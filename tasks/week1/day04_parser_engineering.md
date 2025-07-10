# Day 4: Parser Engineering

## 🎯 Objectives
1. Build robust parsers for structured LLM outputs
2. Implement multiple parsing strategies with fallbacks
3. Create self-healing parsers that learn from failures
4. Design grammar-based parsing for complex structures
5. Build a parsing pipeline with validation and correction

## 📋 Prerequisites
- Completed Days 1-3
- Understanding of parsing concepts (tokenization, AST)
- Familiarity with regular expressions
- Basic knowledge of grammars (optional but helpful)

## 🌄 Morning Tasks (60-75 minutes)

### Task 1: Parser Architecture (25 min)
Create `src/parsing/core.py`:

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TypeVar, Generic, Union, Callable
from enum import Enum
import re
import json
import structlog
from pydantic import BaseModel, ValidationError

logger = structlog.get_logger()

T = TypeVar('T')

class ParseStrategy(Enum):
    """Parsing strategies"""
    JSON = "json"
    XML = "xml"
    MARKDOWN = "markdown"
    CUSTOM_GRAMMAR = "custom_grammar"
    REGEX = "regex"
    NATURAL_LANGUAGE = "natural_language"

@dataclass
class ParseError:
    """Detailed parse error information"""
    strategy: ParseStrategy
    error_type: str
    message: str
    position: Optional[int] = None
    line: Optional[int] = None
    column: Optional[int] = None
    context: Optional[str] = None

@dataclass
class ParseResult(Generic[T]):
    """Result of parsing attempt"""
    success: bool
    data: Optional[T] = None
    errors: List[ParseError] = field(default_factory=list)
    strategy_used: Optional[ParseStrategy] = None
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

class Parser(ABC, Generic[T]):
    """Base parser interface"""
    
    def __init__(self, strategy: ParseStrategy):
        self.strategy = strategy
        self.success_count = 0
        self.failure_count = 0
        
    @abstractmethod
    def parse(self, text: str) -> ParseResult[T]:
        """Parse text and return result"""
        pass
    
    def get_success_rate(self) -> float:
        """Get parser success rate"""
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.0
    
    def record_result(self, success: bool):
        """Record parsing result for statistics"""
        if success:
            self.success_count += 1
        else:
            self.failure_count += 1

class CompositeParser(Parser[T]):
    """Combine multiple parsers with fallback"""
    
    def __init__(self, parsers: List[Parser[T]], strategy: ParseStrategy = ParseStrategy.JSON):
        super().__init__(strategy)
        self.parsers = parsers
        self.parser_weights = {p: 1.0 for p in parsers}
        
    def parse(self, text: str) -> ParseResult[T]:
        """Try each parser in order of success rate"""
        all_errors = []
        
        # Sort parsers by weight (success rate)
        sorted_parsers = sorted(
            self.parsers,
            key=lambda p: self.parser_weights[p],
            reverse=True
        )
        
        for parser in sorted_parsers:
            try:
                result = parser.parse(text)
                
                if result.success:
                    # Update weights based on success
                    self.parser_weights[parser] *= 1.1
                    self.record_result(True)
                    parser.record_result(True)
                    return result
                else:
                    all_errors.extend(result.errors)
                    self.parser_weights[parser] *= 0.9
                    parser.record_result(False)
                    
            except Exception as e:
                all_errors.append(ParseError(
                    strategy=parser.strategy,
                    error_type="exception",
                    message=str(e)
                ))
                self.parser_weights[parser] *= 0.8
                parser.record_result(False)
        
        self.record_result(False)
        return ParseResult(
            success=False,
            errors=all_errors
        )

class ValidationLayer:
    """Validate parsed data against schema"""
    
    def __init__(self, schema: type[BaseModel]):
        self.schema = schema
        
    def validate(self, data: Dict[str, Any]) -> ParseResult:
        """Validate data against schema"""
        try:
            validated = self.schema(**data)
            return ParseResult(
                success=True,
                data=validated.dict(),
                confidence=1.0
            )
        except ValidationError as e:
            errors = []
            for error in e.errors():
                errors.append(ParseError(
                    strategy=ParseStrategy.JSON,
                    error_type="validation",
                    message=error["msg"],
                    context=str(error["loc"])
                ))
            
            return ParseResult(
                success=False,
                errors=errors
            )

class ParseCorrector:
    """Attempt to correct common parsing errors"""
    
    def __init__(self):
        self.corrections = {
            "missing_quotes": self._fix_missing_quotes,
            "trailing_comma": self._fix_trailing_comma,
            "single_quotes": self._fix_single_quotes,
            "unescaped_chars": self._fix_unescaped_chars,
            "incomplete_json": self._fix_incomplete_json,
        }
        
    def correct(self, text: str, error: ParseError) -> Optional[str]:
        """Try to correct text based on error"""
        for error_pattern, correction_func in self.corrections.items():
            if error_pattern in error.message.lower():
                corrected = correction_func(text)
                if corrected != text:
                    logger.info(f"Applied correction: {error_pattern}")
                    return corrected
        
        return None
    
    def _fix_missing_quotes(self, text: str) -> str:
        """Fix missing quotes around keys"""
        # Pattern: key: value -> "key": value
        pattern = r'(\w+):\s*([^,}\]]+)'
        return re.sub(pattern, r'"\1": \2', text)
    
    def _fix_trailing_comma(self, text: str) -> str:
        """Remove trailing commas"""
        # Remove comma before closing brackets
        text = re.sub(r',\s*}', '}', text)
        text = re.sub(r',\s*\]', ']', text)
        return text
    
    def _fix_single_quotes(self, text: str) -> str:
        """Replace single quotes with double quotes"""
        # Be careful not to replace quotes inside strings
        return text.replace("'", '"')
    
    def _fix_unescaped_chars(self, text: str) -> str:
        """Escape special characters"""
        # Escape newlines and tabs in strings
        text = re.sub(r'("(?:[^"\\]|\\.)*")', 
                     lambda m: m.group(1).replace('\n', '\\n').replace('\t', '\\t'),
                     text)
        return text
    
    def _fix_incomplete_json(self, text: str) -> str:
        """Try to complete incomplete JSON"""
        # Count brackets
        open_braces = text.count('{')
        close_braces = text.count('}')
        open_brackets = text.count('[')
        close_brackets = text.count(']')
        
        # Add missing closing brackets
        if open_braces > close_braces:
            text += '}' * (open_braces - close_braces)
        if open_brackets > close_brackets:
            text += ']' * (open_brackets - close_brackets)
            
        return text
```

### Task 2: JSON and XML Parsers (20 min)
Create `src/parsing/json_parser.py`:

```python
import json
import re
from typing import Any, Dict, Optional
import json5  # pip install json5
import hjson  # pip install hjson
import structlog

from src.parsing.core import Parser, ParseResult, ParseError, ParseStrategy

logger = structlog.get_logger()

class SmartJSONParser(Parser[Dict[str, Any]]):
    """Enhanced JSON parser with multiple strategies"""
    
    def __init__(self):
        super().__init__(ParseStrategy.JSON)
        self.strategies = [
            ("standard", self._parse_standard_json),
            ("json5", self._parse_json5),
            ("hjson", self._parse_hjson),
            ("code_block", self._parse_from_code_block),
            ("relaxed", self._parse_relaxed_json),
        ]
        
    def parse(self, text: str) -> ParseResult[Dict[str, Any]]:
        """Try multiple JSON parsing strategies"""
        errors = []
        
        for strategy_name, strategy_func in self.strategies:
            try:
                result = strategy_func(text)
                if result:
                    return ParseResult(
                        success=True,
                        data=result,
                        strategy_used=self.strategy,
                        confidence=self._calculate_confidence(result, strategy_name),
                        metadata={"sub_strategy": strategy_name}
                    )
            except Exception as e:
                errors.append(ParseError(
                    strategy=self.strategy,
                    error_type=f"{strategy_name}_error",
                    message=str(e)
                ))
        
        return ParseResult(
            success=False,
            errors=errors
        )
    
    def _parse_standard_json(self, text: str) -> Optional[Dict[str, Any]]:
        """Standard JSON parsing"""
        return json.loads(text)
    
    def _parse_json5(self, text: str) -> Optional[Dict[str, Any]]:
        """JSON5 parsing (allows comments, trailing commas, etc.)"""
        return json5.loads(text)
    
    def _parse_hjson(self, text: str) -> Optional[Dict[str, Any]]:
        """Human JSON parsing (very relaxed)"""
        return hjson.loads(text)
    
    def _parse_from_code_block(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract and parse JSON from code blocks"""
        patterns = [
            r'```json\s*(.*?)\s*```',
            r'```\s*(.*?)\s*```',
            r'`(.*?)`',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                try:
                    # Try to parse the extracted content
                    for _, parser in self.strategies[:-2]:  # Skip code_block and relaxed
                        try:
                            return parser(match)
                        except:
                            continue
                except:
                    continue
        
        return None
    
    def _parse_relaxed_json(self, text: str) -> Optional[Dict[str, Any]]:
        """Very relaxed JSON parsing with corrections"""
        # Remove comments
        text = re.sub(r'//.*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
        
        # Fix common issues
        text = re.sub(r'(\w+):', r'"\1":', text)  # Quote keys
        text = re.sub(r',\s*}', '}', text)  # Remove trailing commas
        text = re.sub(r',\s*\]', ']', text)
        
        # Try to find JSON-like structure
        json_match = re.search(r'[{\[].*[}\]]', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except:
                # Try with single quotes converted
                try:
                    fixed = json_match.group().replace("'", '"')
                    return json.loads(fixed)
                except:
                    pass
        
        return None
    
    def _calculate_confidence(self, result: Dict[str, Any], strategy: str) -> float:
        """Calculate confidence score for parsed result"""
        if strategy == "standard":
            return 1.0
        elif strategy == "json5":
            return 0.9
        elif strategy == "hjson":
            return 0.8
        elif strategy == "code_block":
            return 0.7
        else:
            return 0.6
```

Create `src/parsing/xml_parser.py`:

```python
import xml.etree.ElementTree as ET
import re
from typing import Any, Dict, Optional
import xmltodict  # pip install xmltodict
import structlog

from src.parsing.core import Parser, ParseResult, ParseError, ParseStrategy

logger = structlog.get_logger()

class XMLStyleParser(Parser[Dict[str, Any]]):
    """Parse XML-style tags from text"""
    
    def __init__(self):
        super().__init__(ParseStrategy.XML)
        
    def parse(self, text: str) -> ParseResult[Dict[str, Any]]:
        """Parse XML-style content"""
        strategies = [
            self._parse_proper_xml,
            self._parse_xml_tags,
            self._parse_pseudo_xml,
        ]
        
        errors = []
        
        for strategy in strategies:
            try:
                result = strategy(text)
                if result:
                    return ParseResult(
                        success=True,
                        data=result,
                        strategy_used=self.strategy,
                        confidence=0.8 if strategy == self._parse_proper_xml else 0.6
                    )
            except Exception as e:
                errors.append(ParseError(
                    strategy=self.strategy,
                    error_type="xml_parse_error",
                    message=str(e)
                ))
        
        return ParseResult(
            success=False,
            errors=errors
        )
    
    def _parse_proper_xml(self, text: str) -> Optional[Dict[str, Any]]:
        """Parse well-formed XML"""
        # Try to find XML content
        xml_match = re.search(r'<\?xml.*?</\w+>', text, re.DOTALL)
        if xml_match:
            return xmltodict.parse(xml_match.group())
        
        # Try without XML declaration
        root_match = re.search(r'<(\w+).*?</\1>', text, re.DOTALL)
        if root_match:
            return xmltodict.parse(root_match.group())
        
        return None
    
    def _parse_xml_tags(self, text: str) -> Optional[Dict[str, Any]]:
        """Parse individual XML-style tags"""
        tag_pattern = r'<(\w+)>(.*?)</\1>'
        matches = re.findall(tag_pattern, text, re.DOTALL)
        
        if not matches:
            return None
        
        result = {}
        for tag, content in matches:
            # Clean content
            content = content.strip()
            
            # Try to parse nested content
            if '<' in content and '>' in content:
                nested = self._parse_xml_tags(content)
                if nested:
                    result[tag] = nested
                else:
                    result[tag] = content
            else:
                result[tag] = content
        
        return result if result else None
    
    def _parse_pseudo_xml(self, text: str) -> Optional[Dict[str, Any]]:
        """Parse pseudo-XML format (more flexible)"""
        # Pattern for <tag>content</tag> or <tag attr="value">content</tag>
        pattern = r'<(\w+)(?:\s+[^>]+)?>(.*?)</\1>|<(\w+)(?:\s+([^>]+))?/>'
        
        result = {}
        
        for match in re.finditer(pattern, text, re.DOTALL):
            if match.group(1):  # Regular tag
                tag = match.group(1)
                content = match.group(2).strip()
                result[tag] = content
            else:  # Self-closing tag
                tag = match.group(3)
                attrs = match.group(4)
                if attrs:
                    # Parse attributes
                    attr_dict = {}
                    for attr_match in re.finditer(r'(\w+)="([^"]*)"', attrs):
                        attr_dict[attr_match.group(1)] = attr_match.group(2)
                    result[tag] = attr_dict
                else:
                    result[tag] = None
        
        return result if result else None
```

### Task 3: Natural Language Parser (20 min)
Create `src/parsing/nl_parser.py`:

```python
import re
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
import structlog
import spacy  # pip install spacy
# python -m spacy download en_core_web_sm

from src.parsing.core import Parser, ParseResult, ParseError, ParseStrategy

logger = structlog.get_logger()

@dataclass
class Intent:
    """Parsed intent from natural language"""
    action: str
    entities: Dict[str, Any]
    confidence: float

class NaturalLanguageParser(Parser[Intent]):
    """Parse natural language into structured intents"""
    
    def __init__(self):
        super().__init__(ParseStrategy.NATURAL_LANGUAGE)
        
        # Load spacy model
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            logger.warning("Spacy model not available, using regex fallback")
            self.nlp = None
        
        # Define patterns for common intents
        self.patterns = {
            "calculate": [
                r"calculate\s+(.+)",
                r"what is\s+(.+)",
                r"compute\s+(.+)",
                r"solve\s+(.+)",
            ],
            "search": [
                r"search for\s+(.+)",
                r"find\s+(.+)",
                r"look for\s+(.+)",
                r"query\s+(.+)",
            ],
            "read_file": [
                r"read (?:the )?file\s+(.+)",
                r"open (?:the )?file\s+(.+)",
                r"load\s+(.+)",
                r"get contents of\s+(.+)",
            ],
            "write_file": [
                r"write (.+) to (?:the )?file (.+)",
                r"save (.+) (?:to|as) (.+)",
                r"create (?:a )?file (.+) with (.+)",
            ],
            "list_files": [
                r"list files in\s+(.+)",
                r"show files in\s+(.+)",
                r"what files are in\s+(.+)",
                r"ls\s+(.+)",
            ],
        }
        
        # Entity extraction patterns
        self.entity_patterns = {
            "number": r'\b\d+(?:\.\d+)?\b',
            "quoted_string": r'"([^"]+)"|\'([^\']+)\'',
            "file_path": r'[\w/\\.-]+\.\w+',
            "url": r'https?://[^\s]+',
            "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        }
    
    def parse(self, text: str) -> ParseResult[Intent]:
        """Parse natural language text into intent"""
        text = text.lower().strip()
        
        # Try spacy parsing first if available
        if self.nlp:
            spacy_result = self._parse_with_spacy(text)
            if spacy_result:
                return ParseResult(
                    success=True,
                    data=spacy_result,
                    strategy_used=self.strategy,
                    confidence=spacy_result.confidence
                )
        
        # Fallback to pattern matching
        pattern_result = self._parse_with_patterns(text)
        if pattern_result:
            return ParseResult(
                success=True,
                data=pattern_result,
                strategy_used=self.strategy,
                confidence=pattern_result.confidence
            )
        
        return ParseResult(
            success=False,
            errors=[ParseError(
                strategy=self.strategy,
                error_type="no_intent_found",
                message="Could not parse intent from text"
            )]
        )
    
    def _parse_with_spacy(self, text: str) -> Optional[Intent]:
        """Parse using spacy NLP"""
        doc = self.nlp(text)
        
        # Extract main verb as action
        action = None
        for token in doc:
            if token.pos_ == "VERB" and not token.is_stop:
                action = token.lemma_
                break
        
        if not action:
            return None
        
        # Extract entities
        entities = {}
        
        # Named entities
        for ent in doc.ents:
            entities[ent.label_.lower()] = ent.text
        
        # Extract numbers
        numbers = []
        for token in doc:
            if token.like_num:
                try:
                    numbers.append(float(token.text))
                except:
                    numbers.append(token.text)
        
        if numbers:
            entities["numbers"] = numbers
        
        # Extract noun phrases as potential arguments
        noun_phrases = [chunk.text for chunk in doc.noun_chunks]
        if noun_phrases:
            entities["arguments"] = noun_phrases
        
        return Intent(
            action=action,
            entities=entities,
            confidence=0.7
        )
    
    def _parse_with_patterns(self, text: str) -> Optional[Intent]:
        """Parse using regex patterns"""
        for intent, patterns in self.patterns.items():
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    entities = self._extract_entities(text, match.groups())
                    
                    return Intent(
                        action=intent,
                        entities=entities,
                        confidence=0.8
                    )
        
        return None
    
    def _extract_entities(self, text: str, groups: Tuple[str, ...]) -> Dict[str, Any]:
        """Extract entities from text"""
        entities = {}
        
        # Add matched groups
        if groups:
            entities["raw_arguments"] = list(groups)
        
        # Extract specific entity types
        for entity_type, pattern in self.entity_patterns.items():
            matches = re.findall(pattern, text)
            if matches:
                if entity_type == "quoted_string":
                    # Handle quoted strings specially
                    entities[entity_type] = [m[0] or m[1] for m in matches if isinstance(m, tuple)]
                else:
                    entities[entity_type] = matches
        
        return entities

class KeyValueParser(Parser[Dict[str, str]]):
    """Parse key-value pairs from text"""
    
    def __init__(self):
        super().__init__(ParseStrategy.CUSTOM_GRAMMAR)
        
    def parse(self, text: str) -> ParseResult[Dict[str, str]]:
        """Parse key-value pairs"""
        patterns = [
            r'(\w+)\s*:\s*([^\n,;]+)',  # key: value
            r'(\w+)\s*=\s*([^\n,;]+)',  # key = value
            r'(\w+)\s*->\s*([^\n,;]+)', # key -> value
        ]
        
        result = {}
        
        for pattern in patterns:
            matches = re.findall(pattern, text)
            for key, value in matches:
                result[key.strip()] = value.strip()
        
        if result:
            return ParseResult(
                success=True,
                data=result,
                strategy_used=self.strategy,
                confidence=0.7
            )
        
        return ParseResult(
            success=False,
            errors=[ParseError(
                strategy=self.strategy,
                error_type="no_pairs_found",
                message="No key-value pairs found"
            )]
        )
```

### Task 4: Grammar-Based Parser (10 min)
Create `src/parsing/grammar_parser.py`:

```python
from typing import Any, Dict, List, Optional, Union
import re
from dataclasses import dataclass
import pyparsing as pp  # pip install pyparsing
import structlog

from src.parsing.core import Parser, ParseResult, ParseError, ParseStrategy

logger = structlog.get_logger()

@dataclass
class Grammar:
    """Define a parsing grammar"""
    name: str
    rules: Dict[str, pp.ParserElement]
    start_rule: str

class GrammarParser(Parser[Any]):
    """Parse using defined grammars"""
    
    def __init__(self, grammar: Grammar):
        super().__init__(ParseStrategy.CUSTOM_GRAMMAR)
        self.grammar = grammar
        
    def parse(self, text: str) -> ParseResult[Any]:
        """Parse text using grammar"""
        try:
            parser = self.grammar.rules[self.grammar.start_rule]
            result = parser.parseString(text, parseAll=True)
            
            return ParseResult(
                success=True,
                data=result.asDict() if hasattr(result, 'asDict') else list(result),
                strategy_used=self.strategy,
                confidence=0.9
            )
        except pp.ParseException as e:
            return ParseResult(
                success=False,
                errors=[ParseError(
                    strategy=self.strategy,
                    error_type="parse_exception",
                    message=str(e),
                    line=e.line,
                    column=e.column,
                    context=e.line if hasattr(e, 'line') else None
                )]
            )

def create_tool_call_grammar() -> Grammar:
    """Create grammar for parsing tool calls"""
    # Define tokens
    identifier = pp.Word(pp.alphas + "_", pp.alphanums + "_")
    string = pp.QuotedString('"') | pp.QuotedString("'")
    number = pp.pyparsing_common.number()
    
    # Define value types
    value = pp.Forward()
    list_value = pp.Suppress("[") + pp.Optional(pp.delimitedList(value)) + pp.Suppress("]")
    dict_value = pp.Suppress("{") + pp.Optional(pp.dictOf(
        string | identifier,
        value
    )) + pp.Suppress("}")
    
    value <<= string | number | list_value | dict_value | identifier
    
    # Define tool call structure
    tool_name = identifier("tool")
    
    # Arguments can be positional or named
    arg = value
    named_arg = identifier + pp.Suppress("=") + value
    args_list = pp.Optional(pp.delimitedList(arg | pp.Group(named_arg)))
    
    tool_call = (
        tool_name + 
        pp.Suppress("(") + 
        args_list("args") + 
        pp.Suppress(")")
    )
    
    return Grammar(
        name="tool_call",
        rules={"tool_call": tool_call},
        start_rule="tool_call"
    )

def create_structured_output_grammar() -> Grammar:
    """Create grammar for structured agent outputs"""
    # Common elements
    string = pp.QuotedString('"') | pp.QuotedString("'")
    number = pp.pyparsing_common.number()
    
    # Thought pattern: "Thought: ..."
    thought = pp.Keyword("Thought:") + pp.restOfLine("content")
    
    # Action pattern: "Action: tool_name(args)"
    tool_name = pp.Word(pp.alphas + "_", pp.alphanums + "_")
    action = pp.Keyword("Action:") + tool_name("tool") + pp.Optional(
        pp.Suppress("(") + pp.SkipTo(")")("args") + pp.Suppress(")")
    )
    
    # Observation pattern: "Observation: ..."
    observation = pp.Keyword("Observation:") + pp.restOfLine("content")
    
    # Final answer pattern: "Final Answer: ..."
    final_answer = pp.Keyword("Final Answer:") + pp.restOfLine("content")
    
    # Complete structure
    agent_output = pp.OneOrMore(
        pp.Group(thought("thought")) |
        pp.Group(action("action")) |
        pp.Group(observation("observation")) |
        pp.Group(final_answer("final_answer"))
    )
    
    return Grammar(
        name="agent_output",
        rules={"output": agent_output},
        start_rule="output"
    )

class MarkdownParser(Parser[Dict[str, Any]]):
    """Parse markdown-formatted content"""
    
    def __init__(self):
        super().__init__(ParseStrategy.MARKDOWN)
        
    def parse(self, text: str) -> ParseResult[Dict[str, Any]]:
        """Parse markdown structure"""
        result = {
            "headers": [],
            "code_blocks": [],
            "lists": [],
            "links": [],
            "sections": {}
        }
        
        # Extract headers
        header_pattern = r'^(#{1,6})\s+(.+)$'
        for match in re.finditer(header_pattern, text, re.MULTILINE):
            level = len(match.group(1))
            title = match.group(2)
            result["headers"].append({
                "level": level,
                "title": title,
                "position": match.start()
            })
        
        # Extract code blocks
        code_pattern = r'```(\w*)\n(.*?)\n```'
        for match in re.finditer(code_pattern, text, re.DOTALL):
            language = match.group(1) or "text"
            code = match.group(2)
            result["code_blocks"].append({
                "language": language,
                "code": code,
                "position": match.start()
            })
        
        # Extract lists
        list_pattern = r'^\s*[-*+]\s+(.+)$'
        current_list = []
        for line in text.split('\n'):
            match = re.match(list_pattern, line)
            if match:
                current_list.append(match.group(1))
            elif current_list:
                result["lists"].append(current_list)
                current_list = []
        
        if current_list:
            result["lists"].append(current_list)
        
        # Extract links
        link_pattern = r'\[([^\]]+)\]\(([^)]+)\)'
        for match in re.finditer(link_pattern, text):
            result["links"].append({
                "text": match.group(1),
                "url": match.group(2)
            })
        
        # Parse sections based on headers
        if result["headers"]:
            for i, header in enumerate(result["headers"]):
                start = header["position"]
                end = result["headers"][i + 1]["position"] if i + 1 < len(result["headers"]) else len(text)
                
                section_text = text[start:end]
                # Remove the header line
                section_text = '\n'.join(section_text.split('\n')[1:])
                
                result["sections"][header["title"]] = section_text.strip()
        
        return ParseResult(
            success=True,
            data=result,
            strategy_used=self.strategy,
            confidence=0.9
        )
```

## 🌞 Afternoon Tasks (90-120 minutes)

### Task 5: Self-Healing Parser System (45 min)
Create `src/parsing/self_healing.py`:

```python
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field
import json
import pickle
from pathlib import Path
import structlog

from src.parsing.core import (
    Parser, ParseResult, ParseError, ParseStrategy,
    CompositeParser, ParseCorrector
)

logger = structlog.get_logger()

@dataclass
class ParseAttempt:
    """Record of a parsing attempt"""
    text: str
    success: bool
    strategy_used: Optional[ParseStrategy]
    errors: List[ParseError]
    correction_applied: Optional[str] = None
    final_result: Optional[Any] = None

class LearningParser(Parser[Any]):
    """Parser that learns from failures"""
    
    def __init__(
        self,
        base_parser: Parser,
        cache_dir: str = ".parser_cache"
    ):
        super().__init__(ParseStrategy.JSON)
        self.base_parser = base_parser
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Learning components
        self.success_cache = {}
        self.failure_patterns = []
        self.corrections = {}
        
        self._load_learning_data()
        
    def parse(self, text: str) -> ParseResult[Any]:
        """Parse with learning"""
        # Check success cache
        text_hash = hash(text)
        if text_hash in self.success_cache:
            logger.info("Using cached parse result")
            return self.success_cache[text_hash]
        
        # Try base parser
        result = self.base_parser.parse(text)
        
        if result.success:
            # Cache success
            self.success_cache[text_hash] = result
            self._save_learning_data()
            return result
        
        # Try learned corrections
        corrected_text = self._apply_learned_corrections(text, result.errors)
        if corrected_text and corrected_text != text:
            logger.info("Applying learned correction")
            result = self.base_parser.parse(corrected_text)
            
            if result.success:
                # Remember this correction
                self.corrections[text_hash] = corrected_text
                self.success_cache[text_hash] = result
                self._save_learning_data()
                return result
        
        # Record failure pattern
        self._record_failure(text, result.errors)
        
        return result
    
    def teach(self, text: str, correct_result: Any):
        """Teach parser the correct result for a text"""
        text_hash = hash(text)
        
        # Create successful parse result
        result = ParseResult(
            success=True,
            data=correct_result,
            strategy_used=self.base_parser.strategy,
            confidence=1.0,
            metadata={"taught": True}
        )
        
        self.success_cache[text_hash] = result
        
        # Try to learn correction pattern
        if text_hash in self.corrections:
            # We already have a correction for this
            pass
        else:
            # Try to find what correction would work
            # This is simplified - in reality would be more sophisticated
            self.corrections[text_hash] = json.dumps(correct_result)
        
        self._save_learning_data()
        logger.info("Parser taught new pattern")
    
    def _apply_learned_corrections(
        self,
        text: str,
        errors: List[ParseError]
    ) -> Optional[str]:
        """Apply corrections learned from past failures"""
        # Direct correction lookup
        text_hash = hash(text)
        if text_hash in self.corrections:
            return self.corrections[text_hash]
        
        # Try pattern-based corrections
        for pattern in self.failure_patterns:
            if self._matches_pattern(text, errors, pattern):
                if pattern.get("correction"):
                    return pattern["correction"](text)
        
        return None
    
    def _record_failure(self, text: str, errors: List[ParseError]):
        """Record failure for learning"""
        pattern = {
            "text_sample": text[:100],  # First 100 chars
            "error_types": [e.error_type for e in errors],
            "error_messages": [e.message for e in errors],
        }
        
        # Don't record duplicates
        for existing in self.failure_patterns:
            if (existing["error_types"] == pattern["error_types"] and
                existing["text_sample"] == pattern["text_sample"]):
                return
        
        self.failure_patterns.append(pattern)
        
        # Keep only recent patterns
        if len(self.failure_patterns) > 100:
            self.failure_patterns = self.failure_patterns[-100:]
        
        self._save_learning_data()
    
    def _matches_pattern(
        self,
        text: str,
        errors: List[ParseError],
        pattern: Dict
    ) -> bool:
        """Check if current failure matches a known pattern"""
        error_types = [e.error_type for e in errors]
        return (
            error_types == pattern["error_types"] and
            text[:100] == pattern["text_sample"]
        )
    
    def _load_learning_data(self):
        """Load learned patterns from disk"""
        cache_file = self.cache_dir / "learning_data.pkl"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    data = pickle.load(f)
                    self.success_cache = data.get("success_cache", {})
                    self.failure_patterns = data.get("failure_patterns", [])
                    self.corrections = data.get("corrections", {})
                logger.info("Loaded learning data",
                          success_cached=len(self.success_cache),
                          failure_patterns=len(self.failure_patterns))
            except Exception as e:
                logger.error(f"Failed to load learning data: {e}")
    
    def _save_learning_data(self):
        """Save learned patterns to disk"""
        cache_file = self.cache_dir / "learning_data.pkl"
        
        try:
            data = {
                "success_cache": self.success_cache,
                "failure_patterns": self.failure_patterns,
                "corrections": self.corrections,
            }
            
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
                
        except Exception as e:
            logger.error(f"Failed to save learning data: {e}")

class AdaptiveParser(Parser[Any]):
    """Parser that adapts strategy based on input characteristics"""
    
    def __init__(self, parsers: Dict[str, Parser]):
        super().__init__(ParseStrategy.JSON)
        self.parsers = parsers
        self.strategy_selector = StrategySelector()
        
    def parse(self, text: str) -> ParseResult[Any]:
        """Parse with adaptive strategy selection"""
        # Analyze text characteristics
        characteristics = self._analyze_text(text)
        
        # Select best strategy
        strategy = self.strategy_selector.select_strategy(characteristics)
        
        if strategy in self.parsers:
            logger.info(f"Selected strategy: {strategy}")
            result = self.parsers[strategy].parse(text)
            
            # Update strategy selector based on result
            self.strategy_selector.update(characteristics, strategy, result.success)
            
            return result
        
        # Fallback to trying all parsers
        return self._try_all_parsers(text)
    
    def _analyze_text(self, text: str) -> Dict[str, Any]:
        """Analyze text characteristics"""
        return {
            "length": len(text),
            "has_braces": '{' in text and '}' in text,
            "has_brackets": '[' in text and ']' in text,
            "has_angle_brackets": '<' in text and '>' in text,
            "has_quotes": '"' in text or "'" in text,
            "has_colons": ':' in text,
            "has_equals": '=' in text,
            "line_count": text.count('\n'),
            "starts_with_brace": text.strip().startswith('{'),
            "starts_with_bracket": text.strip().startswith('['),
            "has_code_block": '```' in text,
        }
    
    def _try_all_parsers(self, text: str) -> ParseResult[Any]:
        """Try all available parsers"""
        errors = []
        
        for name, parser in self.parsers.items():
            try:
                result = parser.parse(text)
                if result.success:
                    return result
                errors.extend(result.errors)
            except Exception as e:
                errors.append(ParseError(
                    strategy=parser.strategy,
                    error_type="exception",
                    message=str(e)
                ))
        
        return ParseResult(success=False, errors=errors)

class StrategySelector:
    """Select parsing strategy based on text characteristics"""
    
    def __init__(self):
        self.strategy_scores = {}
        
    def select_strategy(self, characteristics: Dict[str, Any]) -> str:
        """Select best strategy for given characteristics"""
        scores = {}
        
        # JSON indicators
        json_score = 0
        if characteristics["has_braces"] or characteristics["has_brackets"]:
            json_score += 3
        if characteristics["starts_with_brace"] or characteristics["starts_with_bracket"]:
            json_score += 2
        if characteristics["has_colons"]:
            json_score += 1
        scores["json"] = json_score
        
        # XML indicators
        xml_score = 0
        if characteristics["has_angle_brackets"]:
            xml_score += 3
        if '</>' in str(characteristics.get("text", "")):
            xml_score += 2
        scores["xml"] = xml_score
        
        # Markdown indicators
        md_score = 0
        if characteristics["has_code_block"]:
            md_score += 3
        if characteristics["line_count"] > 5:
            md_score += 1
        scores["markdown"] = md_score
        
        # Natural language indicators
        nl_score = 0
        if not characteristics["has_braces"] and not characteristics["has_angle_brackets"]:
            nl_score += 2
        if characteristics["length"] > 50 and characteristics["line_count"] < 3:
            nl_score += 1
        scores["natural_language"] = nl_score
        
        # Select highest scoring strategy
        return max(scores, key=scores.get)
    
    def update(self, characteristics: Dict[str, Any], strategy: str, success: bool):
        """Update strategy selection based on outcome"""
        key = self._characteristics_key(characteristics)
        
        if key not in self.strategy_scores:
            self.strategy_scores[key] = {}
        
        if strategy not in self.strategy_scores[key]:
            self.strategy_scores[key][strategy] = {"success": 0, "failure": 0}
        
        if success:
            self.strategy_scores[key][strategy]["success"] += 1
        else:
            self.strategy_scores[key][strategy]["failure"] += 1
    
    def _characteristics_key(self, characteristics: Dict[str, Any]) -> str:
        """Create hashable key from characteristics"""
        # Simplified - in reality would be more sophisticated
        key_parts = []
        for k, v in sorted(characteristics.items()):
            if isinstance(v, bool):
                if v:
                    key_parts.append(k)
            elif isinstance(v, (int, float)):
                key_parts.append(f"{k}_{v // 10}")
        
        return "_".join(key_parts)
```

### Task 6: Parser Pipeline (30 min)
Create `src/parsing/pipeline.py`:

```python
from typing import Any, Dict, List, Optional, Callable, Type
from dataclasses import dataclass
import structlog

from src.parsing.core import (
    Parser, ParseResult, ParseError, ParseStrategy,
    ValidationLayer, ParseCorrector
)

logger = structlog.get_logger()

@dataclass
class PipelineStage:
    """Single stage in parsing pipeline"""
    name: str
    processor: Callable[[Any], Any]
    required: bool = True
    error_handler: Optional[Callable[[Exception], Any]] = None

class ParsingPipeline:
    """Complete parsing pipeline with multiple stages"""
    
    def __init__(self):
        self.stages: List[PipelineStage] = []
        self.context: Dict[str, Any] = {}
        
    def add_stage(
        self,
        name: str,
        processor: Callable,
        required: bool = True,
        error_handler: Optional[Callable] = None
    ) -> 'ParsingPipeline':
        """Add a processing stage"""
        self.stages.append(PipelineStage(
            name=name,
            processor=processor,
            required=required,
            error_handler=error_handler
        ))
        return self
    
    def process(self, input_data: Any) -> ParseResult[Any]:
        """Process input through all stages"""
        current_data = input_data
        errors = []
        
        for stage in self.stages:
            try:
                logger.info(f"Processing stage: {stage.name}")
                current_data = stage.processor(current_data)
                self.context[stage.name] = current_data
                
            except Exception as e:
                logger.error(f"Stage {stage.name} failed: {e}")
                
                if stage.error_handler:
                    try:
                        current_data = stage.error_handler(e)
                    except Exception as handler_error:
                        errors.append(ParseError(
                            strategy=ParseStrategy.CUSTOM_GRAMMAR,
                            error_type=f"{stage.name}_handler_error",
                            message=str(handler_error)
                        ))
                        
                        if stage.required:
                            return ParseResult(success=False, errors=errors)
                else:
                    errors.append(ParseError(
                        strategy=ParseStrategy.CUSTOM_GRAMMAR,
                        error_type=f"{stage.name}_error",
                        message=str(e)
                    ))
                    
                    if stage.required:
                        return ParseResult(success=False, errors=errors)
        
        return ParseResult(
            success=True,
            data=current_data,
            errors=errors,
            metadata={"stages_completed": len(self.stages)}
        )

def create_robust_parsing_pipeline(
    parser: Parser,
    validator: Optional[ValidationLayer] = None,
    corrector: Optional[ParseCorrector] = None
) -> ParsingPipeline:
    """Create a complete parsing pipeline"""
    pipeline = ParsingPipeline()
    
    # Stage 1: Pre-processing
    def preprocess(text: str) -> str:
        # Remove zero-width characters
        text = text.replace('\u200b', '').replace('\u200c', '').replace('\u200d', '')
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        # Remove BOM if present
        if text.startswith('\ufeff'):
            text = text[1:]
        
        return text
    
    pipeline.add_stage("preprocess", preprocess, required=False)
    
    # Stage 2: Initial parse attempt
    def initial_parse(text: str) -> ParseResult:
        return parser.parse(text)
    
    pipeline.add_stage("initial_parse", initial_parse)
    
    # Stage 3: Error correction (if initial parse fails)
    if corrector:
        def correct_and_retry(result: ParseResult) -> ParseResult:
            if result.success:
                return result
            
            # Try corrections
            text = pipeline.context.get("preprocess", "")
            for error in result.errors:
                corrected = corrector.correct(text, error)
                if corrected:
                    retry_result = parser.parse(corrected)
                    if retry_result.success:
                        logger.info("Correction successful")
                        return retry_result
            
            return result
        
        pipeline.add_stage("correction", correct_and_retry, required=False)
    
    # Stage 4: Validation (if validator provided)
    if validator:
        def validate(result: ParseResult) -> ParseResult:
            if not result.success:
                return result
            
            return validator.validate(result.data)
        
        pipeline.add_stage("validation", validate)
    
    # Stage 5: Post-processing
    def postprocess(result: ParseResult) -> Any:
        if not result.success:
            return result
        
        # Extract data from result
        return result.data
    
    pipeline.add_stage("postprocess", postprocess)
    
    return pipeline

class StreamingParser:
    """Parse streaming input incrementally"""
    
    def __init__(self, parser: Parser):
        self.parser = parser
        self.buffer = ""
        self.partial_results = []
        
    def feed(self, chunk: str) -> Optional[ParseResult]:
        """Feed a chunk of data"""
        self.buffer += chunk
        
        # Try to parse complete structures
        result = self._try_parse_complete()
        
        if result:
            self.partial_results.append(result)
            return result
        
        return None
    
    def finish(self) -> ParseResult:
        """Finish parsing and return final result"""
        if self.buffer:
            # Try one final parse
            result = self.parser.parse(self.buffer)
            if result.success:
                self.partial_results.append(result)
        
        # Combine all partial results
        if not self.partial_results:
            return ParseResult(
                success=False,
                errors=[ParseError(
                    strategy=self.parser.strategy,
                    error_type="no_complete_structure",
                    message="No complete structure found in stream"
                )]
            )
        
        # Combine data from all results
        combined_data = []
        for result in self.partial_results:
            if result.data:
                combined_data.append(result.data)
        
        return ParseResult(
            success=True,
            data=combined_data if len(combined_data) > 1 else combined_data[0],
            metadata={"chunks_parsed": len(self.partial_results)}
        )
    
    def _try_parse_complete(self) -> Optional[ParseResult]:
        """Try to parse complete structures from buffer"""
        # Simple approach - try to find complete JSON objects
        if self.parser.strategy == ParseStrategy.JSON:
            # Count braces
            brace_count = 0
            for i, char in enumerate(self.buffer):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    
                    if brace_count == 0:
                        # Found complete object
                        complete = self.buffer[:i+1]
                        result = self.parser.parse(complete)
                        
                        if result.success:
                            # Remove parsed portion from buffer
                            self.buffer = self.buffer[i+1:].lstrip()
                            return result
        
        return None
```

### Task 7: Integration and Testing (15 min)
Create `src/parsing/unified_parser.py`:

```python
from typing import Any, Dict, Optional, Type
import structlog

from src.parsing.core import (
    CompositeParser, ValidationLayer, ParseCorrector
)
from src.parsing.json_parser import SmartJSONParser
from src.parsing.xml_parser import XMLStyleParser
from src.parsing.nl_parser import NaturalLanguageParser, KeyValueParser
from src.parsing.grammar_parser import MarkdownParser
from src.parsing.self_healing import LearningParser, AdaptiveParser
from src.parsing.pipeline import create_robust_parsing_pipeline
from pydantic import BaseModel

logger = structlog.get_logger()

class UnifiedParser:
    """Unified parser that combines all parsing strategies"""
    
    def __init__(
        self,
        enable_learning: bool = True,
        enable_correction: bool = True,
        cache_dir: Optional[str] = None
    ):
        # Create individual parsers
        self.json_parser = SmartJSONParser()
        self.xml_parser = XMLStyleParser()
        self.nl_parser = NaturalLanguageParser()
        self.kv_parser = KeyValueParser()
        self.md_parser = MarkdownParser()
        
        # Create composite parser
        self.composite_parser = CompositeParser([
            self.json_parser,
            self.xml_parser,
            self.kv_parser,
        ])
        
        # Create adaptive parser
        self.adaptive_parser = AdaptiveParser({
            "json": self.json_parser,
            "xml": self.xml_parser,
            "natural_language": self.nl_parser,
            "key_value": self.kv_parser,
            "markdown": self.md_parser,
        })
        
        # Wrap with learning if enabled
        if enable_learning:
            self.main_parser = LearningParser(
                self.adaptive_parser,
                cache_dir=cache_dir or ".parser_cache"
            )
        else:
            self.main_parser = self.adaptive_parser
        
        # Create corrector if enabled
        self.corrector = ParseCorrector() if enable_correction else None
        
    def parse(
        self,
        text: str,
        expected_type: Optional[Type[BaseModel]] = None,
        strategy_hint: Optional[str] = None
    ) -> Any:
        """Parse text with all available strategies"""
        # Create validation layer if type provided
        validator = ValidationLayer(expected_type) if expected_type else None
        
        # Create parsing pipeline
        pipeline = create_robust_parsing_pipeline(
            parser=self.main_parser,
            validator=validator,
            corrector=self.corrector
        )
        
        # Process through pipeline
        result = pipeline.process(text)
        
        if result.success:
            return result.data
        else:
            # Log errors and return None
            for error in result.errors:
                logger.error(
                    "Parse error",
                    strategy=error.strategy.value,
                    error_type=error.error_type,
                    message=error.message
                )
            return None
    
    def teach(self, text: str, correct_result: Any):
        """Teach parser the correct result"""
        if hasattr(self.main_parser, 'teach'):
            self.main_parser.teach(text, correct_result)
        else:
            logger.warning("Parser does not support teaching")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get parsing statistics"""
        stats = {
            "adaptive_parser": {
                "strategies": list(self.adaptive_parser.parsers.keys())
            }
        }
        
        # Get individual parser stats
        for name, parser in self.adaptive_parser.parsers.items():
            stats[name] = {
                "success_rate": parser.get_success_rate(),
                "total_attempts": parser.success_count + parser.failure_count
            }
        
        return stats

# Example usage
def example_usage():
    # Create unified parser
    parser = UnifiedParser()
    
    # Parse various formats
    test_cases = [
        # JSON
        '{"action": "calculate", "expression": "2 + 2"}',
        
        # JSON with errors (will be corrected)
        "{action: 'calculate', expression: '2 + 2',}",
        
        # XML style
        "<action>calculate</action><expression>2 + 2</expression>",
        
        # Natural language
        "Please calculate 2 + 2 for me",
        
        # Key-value
        "action: calculate\nexpression: 2 + 2",
        
        # Markdown with code block
        """
        ## Task
        Calculate the following:
        ```
        2 + 2
        ```
        """,
    ]
    
    for text in test_cases:
        print(f"\nParsing: {text[:50]}...")
        result = parser.parse(text)
        print(f"Result: {result}")
    
    # Get statistics
    print("\nParser Statistics:")
    print(parser.get_stats())

if __name__ == "__main__":
    example_usage()
```

## 🌆 Evening Tasks (45-60 minutes)

### Task 8: Parser Testing Suite (20 min)
Create `tests/test_parsers.py`:

```python
import pytest
from typing import Dict, Any

from src.parsing.unified_parser import UnifiedParser
from src.parsing.core import ParseResult
from src.parsing.json_parser import SmartJSONParser
from src.parsing.nl_parser import Intent

class TestUnifiedParser:
    def setup_method(self):
        self.parser = UnifiedParser(enable_learning=False)
    
    def test_json_parsing(self):
        """Test JSON parsing variants"""
        test_cases = [
            # Standard JSON
            ('{"key": "value"}', {"key": "value"}),
            
            # JSON5 style
            ('{key: "value", trailing: "comma",}', {"key": "value", "trailing": "comma"}),
            
            # JSON in code block
            ('```json\n{"key": "value"}\n```', {"key": "value"}),
            
            # Relaxed JSON
            ("{key: 'value'}", {"key": "value"}),
        ]
        
        for text, expected in test_cases:
            result = self.parser.parse(text)
            assert result == expected, f"Failed to parse: {text}"
    
    def test_xml_parsing(self):
        """Test XML-style parsing"""
        text = "<action>calculate</action><input>2+2</input>"
        result = self.parser.parse(text)
        
        assert result is not None
        assert "action" in result
        assert result["action"] == "calculate"
    
    def test_natural_language_parsing(self):
        """Test natural language parsing"""
        # The NL parser returns Intent objects
        # We need to handle this case
        text = "Please calculate 2 + 2"
        result = self.parser.parse(text)
        
        # Result might be an Intent object or dict
        assert result is not None
    
    def test_error_correction(self):
        """Test automatic error correction"""
        parser = UnifiedParser(enable_correction=True)
        
        # Missing quotes
        text = '{key: value}'
        result = parser.parse(text)
        assert result is not None
        
        # Trailing comma
        text = '{"key": "value",}'
        result = parser.parse(text)
        assert result == {"key": "value"}
    
    def test_learning_parser(self):
        """Test learning capabilities"""
        parser = UnifiedParser(enable_learning=True, cache_dir=".test_cache")
        
        # Teach the parser
        difficult_text = "CUSTOM_FORMAT[key=value]"
        correct_result = {"key": "value"}
        
        # First attempt might fail
        result1 = parser.parse(difficult_text)
        
        # Teach correct result
        parser.teach(difficult_text, correct_result)
        
        # Second attempt should succeed
        result2 = parser.parse(difficult_text)
        assert result2 == correct_result

class TestSmartJSONParser:
    def test_json5_features(self):
        parser = SmartJSONParser()
        
        # Comments
        text = '''
        {
            // This is a comment
            "key": "value"
        }
        '''
        result = parser.parse(text)
        assert result.success
        assert result.data["key"] == "value"
    
    def test_code_block_extraction(self):
        parser = SmartJSONParser()
        
        text = '''
        Here's the JSON:
        ```json
        {
            "action": "test",
            "value": 123
        }
        ```
        '''
        result = parser.parse(text)
        assert result.success
        assert result.data["action"] == "test"
        assert result.data["value"] == 123
    
    def test_confidence_scores(self):
        parser = SmartJSONParser()
        
        # Standard JSON should have high confidence
        result1 = parser.parse('{"key": "value"}')
        assert result1.confidence == 1.0
        
        # Relaxed parsing should have lower confidence
        result2 = parser.parse('`{"key": "value"}`')
        assert result2.confidence < 1.0

class TestParsingPipeline:
    def test_pipeline_stages(self):
        from src.parsing.pipeline import ParsingPipeline
        
        pipeline = ParsingPipeline()
        
        # Add test stages
        pipeline.add_stage("double", lambda x: x * 2)
        pipeline.add_stage("add_one", lambda x: x + 1)
        
        result = pipeline.process(5)
        assert result.success
        assert result.data == 11  # (5 * 2) + 1
    
    def test_pipeline_error_handling(self):
        from src.parsing.pipeline import ParsingPipeline
        
        pipeline = ParsingPipeline()
        
        # Add stage that fails
        pipeline.add_stage(
            "fail",
            lambda x: 1 / 0,
            required=False,
            error_handler=lambda e: 0
        )
        
        result = pipeline.process(5)
        assert result.success
        assert result.data == 0  # Error handler returned 0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

### Task 9: Performance Benchmarking (15 min)
Create `benchmarks/benchmark_parsers.py`:

```python
import time
import json
import statistics
from typing import Dict, List, Any

from src.parsing.unified_parser import UnifiedParser
from src.parsing.json_parser import SmartJSONParser
from src.parsing.core import CompositeParser

class ParserBenchmark:
    def __init__(self):
        self.results = {}
        
    def benchmark_parser(
        self,
        parser_name: str,
        parser,
        test_data: List[str],
        iterations: int = 100
    ):
        """Benchmark a parser"""
        times = []
        successes = 0
        
        print(f"\nBenchmarking {parser_name}...")
        
        for _ in range(iterations):
            for data in test_data:
                start = time.time()
                
                try:
                    result = parser.parse(data)
                    if hasattr(result, 'success') and result.success:
                        successes += 1
                    elif result is not None:
                        successes += 1
                except:
                    pass
                
                times.append(time.time() - start)
        
        self.results[parser_name] = {
            "total_time": sum(times),
            "avg_time": statistics.mean(times),
            "min_time": min(times),
            "max_time": max(times),
            "median_time": statistics.median(times),
            "success_rate": successes / (iterations * len(test_data)),
            "operations_per_second": len(times) / sum(times)
        }
    
    def generate_test_data(self) -> Dict[str, List[str]]:
        """Generate test data for different formats"""
        return {
            "simple_json": [
                '{"key": "value"}',
                '{"number": 123}',
                '{"bool": true}',
                '{"array": [1, 2, 3]}',
                '{"nested": {"key": "value"}}',
            ],
            "complex_json": [
                '{"users": [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]}',
                '{"config": {"server": {"host": "localhost", "port": 8080}}}',
                json.dumps({"data": [{"x": i, "y": i**2} for i in range(10)]}),
            ],
            "malformed_json": [
                '{key: "value"}',  # Missing quotes
                '{"key": "value",}',  # Trailing comma
                "{'key': 'value'}",  # Single quotes
                '{"key": "value"',  # Incomplete
            ],
            "mixed_formats": [
                "<action>test</action>",
                "key: value",
                "Calculate 2 + 2",
                "```json\n{\"key\": \"value\"}\n```",
            ]
        }
    
    def run_benchmarks(self):
        """Run all benchmarks"""
        # Create parsers
        parsers = {
            "UnifiedParser": UnifiedParser(enable_learning=False),
            "SmartJSONParser": SmartJSONParser(),
            "UnifiedParser_with_learning": UnifiedParser(enable_learning=True),
        }
        
        # Generate test data
        test_data_sets = self.generate_test_data()
        
        # Run benchmarks for each parser and data set
        for parser_name, parser in parsers.items():
            all_data = []
            for data_set in test_data_sets.values():
                all_data.extend(data_set)
            
            self.benchmark_parser(parser_name, parser, all_data)
        
        # Print results
        self.print_results()
    
    def print_results(self):
        """Print benchmark results"""
        print("\n" + "="*60)
        print("PARSER BENCHMARK RESULTS")
        print("="*60)
        
        for parser_name, results in self.results.items():
            print(f"\n{parser_name}:")
            print(f"  Average time: {results['avg_time']*1000:.3f} ms")
            print(f"  Min time: {results['min_time']*1000:.3f} ms")
            print(f"  Max time: {results['max_time']*1000:.3f} ms")
            print(f"  Median time: {results['median_time']*1000:.3f} ms")
            print(f"  Success rate: {results['success_rate']*100:.1f}%")
            print(f"  Ops/second: {results['operations_per_second']:.1f}")

if __name__ == "__main__":
    benchmark = ParserBenchmark()
    benchmark.run_benchmarks()
```

### Task 10: Update Learning Journal (10 min)

Update your CLAUDE.md:

```markdown
## Day 4: Parser Engineering

### What I Built
- ✅ Morning: Core parser architecture with multiple strategies
- ✅ Afternoon: Self-healing and adaptive parsers
- ✅ Evening: Complete parsing pipeline with testing

### Key Learnings
1. **Technical**: 
   - Multiple parsing strategies increase robustness
   - Self-correction can handle common LLM output issues
   - Learning from failures improves over time

2. **Architecture**:
   - Pipeline pattern allows staged processing
   - Strategy pattern enables parser selection
   - Composite pattern for fallback parsing

3. **Performance**:
   - Caching successful parses saves time
   - Adaptive strategy selection improves efficiency
   - Streaming parsing enables real-time processing

### Challenges Faced
- **Issue**: LLMs produce inconsistent output formats
  **Solution**: Multiple parsing strategies with fallback
  **Lesson**: Never assume output format consistency

- **Issue**: Error correction can introduce new errors
  **Solution**: Validate corrections before accepting
  **Lesson**: Corrections need constraints

### Code Metrics
- Lines written: ~2000
- Test cases: 25
- Parsing strategies: 6
- Success rate: 95%+ on test data

### Tomorrow's Goal
- [ ] Implement state machines for agents
- [ ] Build conversation memory system
- [ ] Add context management
```

## 📊 Deliverables Checklist
- [ ] Core parser architecture
- [ ] JSON and XML parsers with fallbacks
- [ ] Natural language parser
- [ ] Grammar-based parsing
- [ ] Self-healing parser system
- [ ] Complete parsing pipeline
- [ ] Comprehensive test suite
- [ ] Performance benchmarks

## 🎯 Success Metrics
You've succeeded if you can:
1. Parse 5+ different output formats correctly
2. Automatically correct common parsing errors
3. Learn from parsing failures
4. Handle streaming input
5. Achieve 90%+ success rate on mixed formats

## 🚀 Extension Challenges
If you finish early:
1. Add parser for YAML and TOML formats
2. Implement fuzzy matching for partial parses
3. Build visual parser debugger
4. Add parser performance profiling
5. Create DSL for custom grammar definition

---

🎉 **Congratulations!** You've built a sophisticated parsing system that can handle real-world LLM outputs. Tomorrow we'll focus on state management and memory systems.