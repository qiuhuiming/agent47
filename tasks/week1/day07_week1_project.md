# Day 7: Week 1 Project - File System Analyzer Agent

## 🎯 Project Overview
Build a production-ready AI agent that can analyze codebases, understand project structure, answer questions about code, and provide insights. This project combines everything learned this week.

## 📋 Project Requirements
1. **Core Features**:
   - Analyze file structure and dependencies
   - Answer questions about code functionality
   - Generate documentation
   - Find code patterns and potential issues
   - Provide refactoring suggestions

2. **Technical Requirements**:
   - Use advanced LLM client with streaming
   - Implement robust parsing for various outputs
   - Use state machines for agent logic
   - Apply async patterns for performance
   - Include comprehensive error handling

3. **Production Features**:
   - Configuration management
   - Logging and monitoring
   - Cost tracking
   - Performance optimization
   - Testing suite

## 🌄 Morning Tasks (60-90 minutes)

### Task 1: Project Setup and Architecture (30 min)
Create the project structure:

```bash
mkdir -p file_analyzer_agent/{src,tests,tools,config,logs,docs}
cd file_analyzer_agent
```

Create `src/config/settings.py`:

```python
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict
import os
import yaml
from dotenv import load_dotenv

load_dotenv()

@dataclass
class AnalyzerConfig:
    """Configuration for file analyzer agent"""
    # Paths
    project_root: Path
    cache_dir: Path = Path(".cache")
    log_dir: Path = Path("logs")
    
    # LLM settings
    llm_provider: str = "openai"
    llm_model: str = "gpt-3.5-turbo"
    llm_temperature: float = 0.3
    llm_max_tokens: int = 2000
    
    # Analysis settings
    max_file_size_mb: float = 10.0
    ignored_patterns: List[str] = None
    file_extensions: List[str] = None
    max_files_to_analyze: int = 1000
    
    # Performance settings
    max_concurrent_files: int = 10
    max_concurrent_tools: int = 5
    enable_caching: bool = True
    cache_ttl_hours: int = 24
    
    # Cost settings
    daily_cost_limit: float = 10.0
    cost_per_1k_tokens: float = 0.002
    
    def __post_init__(self):
        # Set defaults
        if self.ignored_patterns is None:
            self.ignored_patterns = [
                "__pycache__", ".git", ".venv", "venv",
                "node_modules", ".pytest_cache", "*.pyc"
            ]
        
        if self.file_extensions is None:
            self.file_extensions = [
                ".py", ".js", ".ts", ".java", ".cpp", ".c",
                ".go", ".rs", ".rb", ".php", ".swift"
            ]
        
        # Create directories
        self.cache_dir.mkdir(exist_ok=True)
        self.log_dir.mkdir(exist_ok=True)
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'AnalyzerConfig':
        """Load config from YAML file"""
        with open(config_path, 'r') as f:
            data = yaml.safe_load(f)
        
        # Override with environment variables
        data['llm_provider'] = os.getenv('LLM_PROVIDER', data.get('llm_provider', 'openai'))
        data['project_root'] = Path(data.get('project_root', '.'))
        
        return cls(**data)
    
    @classmethod
    def from_env(cls, project_root: str = ".") -> 'AnalyzerConfig':
        """Create config from environment"""
        return cls(
            project_root=Path(project_root),
            llm_provider=os.getenv('LLM_PROVIDER', 'openai'),
            llm_model=os.getenv('LLM_MODEL', 'gpt-3.5-turbo'),
            daily_cost_limit=float(os.getenv('DAILY_COST_LIMIT', '10.0'))
        )
```

Create `src/core/analyzer_agent.py`:

```python
import asyncio
from typing import Dict, List, Optional, Any
from pathlib import Path
import structlog

from src.config.settings import AnalyzerConfig
from src.state.analyzer_states import AnalyzerStateMachine
from src.tools.code_analysis_tools import CodeAnalysisToolkit
from src.llm.client import create_llm_client
from src.parsing.unified_parser import UnifiedParser
from src.async_patterns.core import AsyncPipeline
from src.llm.cost_tracker import CostTracker

logger = structlog.get_logger()

class FileAnalyzerAgent:
    """Main file analyzer agent"""
    
    def __init__(self, config: AnalyzerConfig):
        self.config = config
        
        # Initialize components
        self.llm = create_llm_client(config)
        self.parser = UnifiedParser(enable_learning=True)
        self.cost_tracker = CostTracker(str(config.log_dir / "usage.json"))
        
        # Initialize tools
        self.toolkit = CodeAnalysisToolkit(config)
        
        # Initialize state machine
        self.state_machine = AnalyzerStateMachine(
            agent=self,
            config=config
        )
        
        # Analysis cache
        self.analysis_cache: Dict[str, Any] = {}
        
        # Setup pipeline
        self.pipeline = AsyncPipeline()
        self._setup_pipeline()
        
        self.initialized = False
    
    async def initialize(self):
        """Initialize the agent"""
        if self.initialized:
            return
        
        logger.info("Initializing file analyzer agent")
        
        # Load project structure
        await self._load_project_structure()
        
        # Initialize state machine
        await self.state_machine.initialize()
        
        # Start background tasks
        asyncio.create_task(self._monitor_costs())
        
        self.initialized = True
        logger.info("File analyzer agent initialized")
    
    async def analyze_project(self) -> Dict[str, Any]:
        """Perform complete project analysis"""
        logger.info(f"Starting project analysis: {self.config.project_root}")
        
        # Run analysis pipeline
        result = await self.pipeline.execute({
            "project_root": self.config.project_root,
            "config": self.config
        })
        
        # Cache result
        self.analysis_cache["full_analysis"] = result
        
        return result
    
    async def answer_question(self, question: str) -> str:
        """Answer a question about the codebase"""
        # Ensure we have analysis
        if "full_analysis" not in self.analysis_cache:
            await self.analyze_project()
        
        # Use state machine for question answering
        context = {
            "question": question,
            "analysis": self.analysis_cache["full_analysis"]
        }
        
        result = await self.state_machine.run(
            initial_state="question_received",
            context=context
        )
        
        return result.get("answer", "I couldn't find an answer to that question.")
    
    async def find_pattern(self, pattern: str) -> List[Dict[str, Any]]:
        """Find a pattern in the codebase"""
        results = await self.toolkit.pattern_finder.find_pattern(pattern)
        return results
    
    async def generate_documentation(
        self,
        output_format: str = "markdown"
    ) -> str:
        """Generate project documentation"""
        if "full_analysis" not in self.analysis_cache:
            await self.analyze_project()
        
        doc_generator = self.toolkit.doc_generator
        return await doc_generator.generate(
            self.analysis_cache["full_analysis"],
            format=output_format
        )
    
    def _setup_pipeline(self):
        """Setup analysis pipeline"""
        self.pipeline.add_stage(self._scan_files)
        self.pipeline.add_stage(self._analyze_structure)
        self.pipeline.add_stage(self._extract_metadata)
        self.pipeline.add_stage(self._analyze_dependencies)
        self.pipeline.add_stage(self._identify_patterns)
        self.pipeline.add_stage(self._generate_insights)
    
    async def _load_project_structure(self):
        """Load initial project structure"""
        structure = await self.toolkit.file_scanner.scan_directory(
            self.config.project_root
        )
        self.analysis_cache["structure"] = structure
    
    async def _monitor_costs(self):
        """Monitor API costs"""
        while True:
            await asyncio.sleep(300)  # Check every 5 minutes
            
            summary = self.cost_tracker.get_summary(period_days=1)
            daily_cost = summary.get("total_cost", 0)
            
            if daily_cost > self.config.daily_cost_limit:
                logger.warning(
                    f"Daily cost limit exceeded: ${daily_cost:.2f}",
                    limit=self.config.daily_cost_limit
                )
                # Could implement throttling here
    
    # Pipeline stages
    async def _scan_files(self, context: Dict) -> Dict:
        """Scan project files"""
        files = await self.toolkit.file_scanner.scan_directory(
            context["project_root"],
            extensions=self.config.file_extensions,
            ignore_patterns=self.config.ignored_patterns
        )
        context["files"] = files
        logger.info(f"Scanned {len(files)} files")
        return context
    
    async def _analyze_structure(self, context: Dict) -> Dict:
        """Analyze project structure"""
        analyzer = self.toolkit.structure_analyzer
        structure = await analyzer.analyze(context["files"])
        context["structure"] = structure
        return context
    
    async def _extract_metadata(self, context: Dict) -> Dict:
        """Extract metadata from files"""
        extractor = self.toolkit.metadata_extractor
        metadata = await extractor.extract_batch(
            context["files"],
            max_concurrent=self.config.max_concurrent_files
        )
        context["metadata"] = metadata
        return context
    
    async def _analyze_dependencies(self, context: Dict) -> Dict:
        """Analyze code dependencies"""
        analyzer = self.toolkit.dependency_analyzer
        dependencies = await analyzer.analyze(
            context["files"],
            context["metadata"]
        )
        context["dependencies"] = dependencies
        return context
    
    async def _identify_patterns(self, context: Dict) -> Dict:
        """Identify code patterns"""
        identifier = self.toolkit.pattern_identifier
        patterns = await identifier.identify(
            context["files"],
            context["metadata"]
        )
        context["patterns"] = patterns
        return context
    
    async def _generate_insights(self, context: Dict) -> Dict:
        """Generate insights using LLM"""
        # Prepare summary for LLM
        summary = self._prepare_analysis_summary(context)
        
        prompt = f"""
Analyze this codebase and provide insights:

{summary}

Provide:
1. Overview of the project
2. Key architectural patterns
3. Potential improvements
4. Code quality assessment
"""
        
        insights = await self.llm.complete(prompt)
        context["insights"] = self.parser.parse(insights)
        
        return context
    
    def _prepare_analysis_summary(self, context: Dict) -> str:
        """Prepare analysis summary for LLM"""
        summary_parts = []
        
        # File statistics
        files = context.get("files", [])
        summary_parts.append(f"Total files: {len(files)}")
        
        # Structure summary
        if "structure" in context:
            structure = context["structure"]
            summary_parts.append(f"Directories: {structure.get('directory_count', 0)}")
            summary_parts.append(f"Max depth: {structure.get('max_depth', 0)}")
        
        # Language distribution
        if "metadata" in context:
            languages = {}
            for file_meta in context["metadata"].values():
                lang = file_meta.get("language", "unknown")
                languages[lang] = languages.get(lang, 0) + 1
            
            summary_parts.append("\nLanguages:")
            for lang, count in sorted(languages.items(), key=lambda x: x[1], reverse=True):
                summary_parts.append(f"  {lang}: {count} files")
        
        # Dependencies
        if "dependencies" in context:
            deps = context["dependencies"]
            summary_parts.append(f"\nTotal dependencies: {len(deps.get('external', []))}")
        
        return "\n".join(summary_parts)
    
    def get_status(self) -> Dict[str, Any]:
        """Get agent status"""
        return {
            "initialized": self.initialized,
            "cached_analyses": list(self.analysis_cache.keys()),
            "state_machine_state": self.state_machine.current_state,
            "cost_today": self.cost_tracker.get_summary(period_days=1).get("total_cost", 0)
        }
```

### Task 2: Code Analysis Tools (30 min)
Create `src/tools/code_analysis_tools.py`:

```python
import ast
import asyncio
from typing import Dict, List, Any, Optional
from pathlib import Path
import re
import aiofiles
import structlog

from src.tools.base import Tool, ToolResult
from src.config.settings import AnalyzerConfig
from src.async_patterns.concurrent_tools import ConcurrentToolExecutor

logger = structlog.get_logger()

class FileScanner(Tool):
    """Scan directory for code files"""
    
    def __init__(self, config: AnalyzerConfig):
        super().__init__(
            name="file_scanner",
            description="Scan directory for code files"
        )
        self.config = config
    
    async def execute(self, input_data: Dict) -> ToolResult:
        """Execute file scanning"""
        directory = Path(input_data.get("directory", self.config.project_root))
        extensions = input_data.get("extensions", self.config.file_extensions)
        ignore_patterns = input_data.get("ignore_patterns", self.config.ignored_patterns)
        
        files = await self.scan_directory(directory, extensions, ignore_patterns)
        
        return ToolResult(
            success=True,
            output=f"Found {len(files)} files",
            result=files
        )
    
    async def scan_directory(
        self,
        directory: Path,
        extensions: List[str] = None,
        ignore_patterns: List[str] = None
    ) -> List[Dict[str, Any]]:
        """Scan directory for files"""
        files = []
        
        for path in directory.rglob("*"):
            # Skip ignored patterns
            if ignore_patterns:
                if any(pattern in str(path) for pattern in ignore_patterns):
                    continue
            
            # Check file extension
            if path.is_file():
                if extensions and path.suffix not in extensions:
                    continue
                
                # Get file info
                stat = path.stat()
                
                # Skip large files
                if stat.st_size > self.config.max_file_size_mb * 1024 * 1024:
                    logger.warning(f"Skipping large file: {path}")
                    continue
                
                files.append({
                    "path": str(path),
                    "relative_path": str(path.relative_to(directory)),
                    "name": path.name,
                    "extension": path.suffix,
                    "size": stat.st_size,
                    "modified": stat.st_mtime
                })
        
        # Limit number of files
        if len(files) > self.config.max_files_to_analyze:
            logger.warning(f"Limiting to {self.config.max_files_to_analyze} files")
            files = files[:self.config.max_files_to_analyze]
        
        return files

class CodeParser(Tool):
    """Parse code files to extract structure"""
    
    def __init__(self):
        super().__init__(
            name="code_parser",
            description="Parse code to extract structure"
        )
    
    async def execute(self, input_data: Dict) -> ToolResult:
        """Execute code parsing"""
        file_path = input_data.get("file_path")
        language = input_data.get("language", "python")
        
        if not file_path:
            return ToolResult(success=False, error="No file path provided")
        
        try:
            async with aiofiles.open(file_path, 'r') as f:
                content = await f.read()
            
            if language == "python":
                structure = await self.parse_python(content)
            else:
                structure = await self.parse_generic(content)
            
            return ToolResult(
                success=True,
                output="Parsed successfully",
                result=structure
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Parse error: {str(e)}"
            )
    
    async def parse_python(self, content: str) -> Dict[str, Any]:
        """Parse Python code"""
        try:
            tree = ast.parse(content)
            
            structure = {
                "imports": [],
                "classes": [],
                "functions": [],
                "variables": []
            }
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        structure["imports"].append(alias.name)
                        
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    for alias in node.names:
                        structure["imports"].append(f"{module}.{alias.name}")
                        
                elif isinstance(node, ast.ClassDef):
                    structure["classes"].append({
                        "name": node.name,
                        "line": node.lineno,
                        "methods": [
                            m.name for m in node.body
                            if isinstance(m, ast.FunctionDef)
                        ]
                    })
                    
                elif isinstance(node, ast.FunctionDef):
                    # Only top-level functions
                    if not any(isinstance(p, ast.ClassDef) for p in ast.walk(tree) if node in ast.walk(p)):
                        structure["functions"].append({
                            "name": node.name,
                            "line": node.lineno,
                            "args": [arg.arg for arg in node.args.args]
                        })
            
            return structure
            
        except SyntaxError as e:
            return {"error": f"Syntax error: {str(e)}"}
    
    async def parse_generic(self, content: str) -> Dict[str, Any]:
        """Generic parsing for other languages"""
        lines = content.split('\n')
        
        structure = {
            "line_count": len(lines),
            "imports": [],
            "functions": [],
            "classes": []
        }
        
        # Simple regex-based parsing
        import_patterns = [
            r'import\s+(\S+)',
            r'from\s+(\S+)\s+import',
            r'require\s*\(\s*["\']([^"\']+)["\']\s*\)',
            r'#include\s*[<"]([^>"]+)[>"]'
        ]
        
        function_patterns = [
            r'def\s+(\w+)\s*\(',
            r'function\s+(\w+)\s*\(',
            r'func\s+(\w+)\s*\(',
            r'(\w+)\s*:\s*function\s*\(',
        ]
        
        class_patterns = [
            r'class\s+(\w+)',
            r'struct\s+(\w+)',
            r'interface\s+(\w+)',
        ]
        
        for i, line in enumerate(lines):
            # Check imports
            for pattern in import_patterns:
                match = re.search(pattern, line)
                if match:
                    structure["imports"].append(match.group(1))
            
            # Check functions
            for pattern in function_patterns:
                match = re.search(pattern, line)
                if match:
                    structure["functions"].append({
                        "name": match.group(1),
                        "line": i + 1
                    })
            
            # Check classes
            for pattern in class_patterns:
                match = re.search(pattern, line)
                if match:
                    structure["classes"].append({
                        "name": match.group(1),
                        "line": i + 1
                    })
        
        return structure

class DependencyAnalyzer(Tool):
    """Analyze code dependencies"""
    
    def __init__(self):
        super().__init__(
            name="dependency_analyzer",
            description="Analyze code dependencies"
        )
    
    async def execute(self, input_data: Dict) -> ToolResult:
        """Execute dependency analysis"""
        files = input_data.get("files", [])
        metadata = input_data.get("metadata", {})
        
        dependencies = await self.analyze(files, metadata)
        
        return ToolResult(
            success=True,
            output=f"Found {len(dependencies['external'])} external dependencies",
            result=dependencies
        )
    
    async def analyze(
        self,
        files: List[Dict],
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze dependencies"""
        dependencies = {
            "external": set(),
            "internal": {},
            "circular": []
        }
        
        # Extract imports from metadata
        for file_path, file_meta in metadata.items():
            if "structure" in file_meta and "imports" in file_meta["structure"]:
                imports = file_meta["structure"]["imports"]
                
                for imp in imports:
                    # Classify as internal or external
                    if self._is_internal_import(imp, files):
                        if file_path not in dependencies["internal"]:
                            dependencies["internal"][file_path] = []
                        dependencies["internal"][file_path].append(imp)
                    else:
                        dependencies["external"].add(imp)
        
        # Detect circular dependencies
        dependencies["circular"] = self._detect_circular_deps(
            dependencies["internal"]
        )
        
        # Convert set to list for JSON serialization
        dependencies["external"] = list(dependencies["external"])
        
        return dependencies
    
    def _is_internal_import(self, import_name: str, files: List[Dict]) -> bool:
        """Check if import is internal to project"""
        # Simple heuristic - check if any file matches import
        for file in files:
            if import_name in file["relative_path"]:
                return True
        return False
    
    def _detect_circular_deps(self, internal_deps: Dict[str, List[str]]) -> List[List[str]]:
        """Detect circular dependencies"""
        circular = []
        
        def find_cycles(node: str, path: List[str], visited: set):
            if node in path:
                # Found cycle
                cycle_start = path.index(node)
                cycle = path[cycle_start:] + [node]
                circular.append(cycle)
                return
            
            if node in visited:
                return
            
            visited.add(node)
            path.append(node)
            
            # Check dependencies
            if node in internal_deps:
                for dep in internal_deps[node]:
                    # Find file that provides this import
                    for file in internal_deps:
                        if dep in file:
                            find_cycles(file, path.copy(), visited.copy())
        
        # Check each file
        for file in internal_deps:
            find_cycles(file, [], set())
        
        return circular

class PatternFinder(Tool):
    """Find patterns in code"""
    
    def __init__(self):
        super().__init__(
            name="pattern_finder",
            description="Find patterns in code"
        )
        
        self.patterns = {
            "todo": r'#\s*TODO:?\s*(.*)',
            "fixme": r'#\s*FIXME:?\s*(.*)',
            "hack": r'#\s*HACK:?\s*(.*)',
            "deprecated": r'@deprecated|#\s*deprecated',
            "security": r'(password|secret|key|token)\s*=\s*["\'][^"\']+["\']',
            "print_debug": r'print\s*\(|console\.log\s*\(',
            "exception_catch_all": r'except\s*:|\bcatch\s*\(',
            "hardcoded_values": r'(localhost|127\.0\.0\.1|8080|3000)',
        }
    
    async def execute(self, input_data: Dict) -> ToolResult:
        """Execute pattern finding"""
        pattern_name = input_data.get("pattern")
        files = input_data.get("files", [])
        
        if pattern_name and pattern_name in self.patterns:
            results = await self.find_pattern(
                self.patterns[pattern_name],
                files
            )
        else:
            # Find all patterns
            results = {}
            for name, pattern in self.patterns.items():
                matches = await self.find_pattern(pattern, files)
                if matches:
                    results[name] = matches
        
        return ToolResult(
            success=True,
            output=f"Found patterns in {len(results)} categories",
            result=results
        )
    
    async def find_pattern(
        self,
        pattern: str,
        files: Optional[List[Dict]] = None
    ) -> List[Dict[str, Any]]:
        """Find pattern in files"""
        results = []
        regex = re.compile(pattern, re.IGNORECASE)
        
        # Get files if not provided
        if not files:
            # Would need to scan files
            return results
        
        # Search in files
        tasks = []
        for file in files:
            task = self._search_file(file["path"], regex)
            tasks.append(task)
        
        # Limit concurrency
        from src.async_patterns.optimization import bounded_gather
        file_results = await bounded_gather(*tasks, limit=10)
        
        # Combine results
        for file, matches in zip(files, file_results):
            if matches:
                results.extend([
                    {
                        "file": file["relative_path"],
                        "line": line_num,
                        "match": match,
                        "pattern": pattern
                    }
                    for line_num, match in matches
                ])
        
        return results
    
    async def _search_file(
        self,
        file_path: str,
        regex: re.Pattern
    ) -> List[tuple]:
        """Search pattern in single file"""
        matches = []
        
        try:
            async with aiofiles.open(file_path, 'r') as f:
                lines = await f.readlines()
                
            for i, line in enumerate(lines, 1):
                match = regex.search(line)
                if match:
                    matches.append((i, match.group(0)))
                    
        except Exception as e:
            logger.error(f"Error searching file {file_path}: {e}")
        
        return matches

class CodeAnalysisToolkit:
    """Collection of code analysis tools"""
    
    def __init__(self, config: AnalyzerConfig):
        self.config = config
        
        # Initialize tools
        self.file_scanner = FileScanner(config)
        self.code_parser = CodeParser()
        self.dependency_analyzer = DependencyAnalyzer()
        self.pattern_finder = PatternFinder()
        
        # Create executor for concurrent execution
        self.executor = ConcurrentToolExecutor(
            tools={
                "file_scanner": self.file_scanner,
                "code_parser": self.code_parser,
                "dependency_analyzer": self.dependency_analyzer,
                "pattern_finder": self.pattern_finder,
            },
            max_concurrent=config.max_concurrent_tools
        )
        
        # Additional analyzers
        self.structure_analyzer = StructureAnalyzer()
        self.metadata_extractor = MetadataExtractor(self.code_parser)
        self.pattern_identifier = PatternIdentifier()
        self.doc_generator = DocumentationGenerator()

class StructureAnalyzer:
    """Analyze project structure"""
    
    async def analyze(self, files: List[Dict]) -> Dict[str, Any]:
        """Analyze structure"""
        structure = {
            "directory_count": 0,
            "max_depth": 0,
            "file_distribution": {},
            "largest_directories": []
        }
        
        directories = {}
        
        for file in files:
            path = Path(file["relative_path"])
            
            # Count directories
            for parent in path.parents:
                directories[str(parent)] = directories.get(str(parent), 0) + 1
            
            # Track depth
            depth = len(path.parts) - 1
            structure["max_depth"] = max(structure["max_depth"], depth)
            
            # File distribution by extension
            ext = file["extension"]
            structure["file_distribution"][ext] = \
                structure["file_distribution"].get(ext, 0) + 1
        
        structure["directory_count"] = len(directories)
        
        # Find largest directories
        sorted_dirs = sorted(
            directories.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        structure["largest_directories"] = [
            {"path": path, "file_count": count}
            for path, count in sorted_dirs
        ]
        
        return structure

class MetadataExtractor:
    """Extract metadata from files"""
    
    def __init__(self, parser: CodeParser):
        self.parser = parser
    
    async def extract_batch(
        self,
        files: List[Dict],
        max_concurrent: int = 10
    ) -> Dict[str, Any]:
        """Extract metadata from multiple files"""
        tasks = []
        
        for file in files:
            task = self.extract_single(file)
            tasks.append(task)
        
        # Execute with concurrency limit
        from src.async_patterns.optimization import bounded_gather
        results = await bounded_gather(*tasks, limit=max_concurrent)
        
        # Create metadata dict
        metadata = {}
        for file, meta in zip(files, results):
            if meta:
                metadata[file["path"]] = meta
        
        return metadata
    
    async def extract_single(self, file: Dict) -> Optional[Dict[str, Any]]:
        """Extract metadata from single file"""
        try:
            # Determine language
            language = self._detect_language(file["extension"])
            
            # Parse structure
            parse_result = await self.parser.execute({
                "file_path": file["path"],
                "language": language
            })
            
            if parse_result.success:
                return {
                    "language": language,
                    "size": file["size"],
                    "structure": parse_result.result,
                    "metrics": await self._calculate_metrics(file["path"])
                }
                
        except Exception as e:
            logger.error(f"Error extracting metadata from {file['path']}: {e}")
        
        return None
    
    def _detect_language(self, extension: str) -> str:
        """Detect language from extension"""
        mapping = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".java": "java",
            ".cpp": "cpp",
            ".c": "c",
            ".go": "go",
            ".rs": "rust",
            ".rb": "ruby",
            ".php": "php"
        }
        return mapping.get(extension, "unknown")
    
    async def _calculate_metrics(self, file_path: str) -> Dict[str, int]:
        """Calculate code metrics"""
        metrics = {
            "lines": 0,
            "code_lines": 0,
            "comment_lines": 0,
            "blank_lines": 0
        }
        
        try:
            async with aiofiles.open(file_path, 'r') as f:
                lines = await f.readlines()
            
            for line in lines:
                line = line.strip()
                metrics["lines"] += 1
                
                if not line:
                    metrics["blank_lines"] += 1
                elif line.startswith(('#', '//', '/*', '*')):
                    metrics["comment_lines"] += 1
                else:
                    metrics["code_lines"] += 1
                    
        except Exception:
            pass
        
        return metrics

class PatternIdentifier:
    """Identify design patterns"""
    
    async def identify(
        self,
        files: List[Dict],
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Identify patterns"""
        patterns = {
            "design_patterns": [],
            "anti_patterns": [],
            "code_smells": []
        }
        
        # Check for common patterns
        for file_path, meta in metadata.items():
            if "structure" not in meta:
                continue
                
            structure = meta["structure"]
            
            # Singleton pattern
            if self._is_singleton(structure):
                patterns["design_patterns"].append({
                    "pattern": "singleton",
                    "file": file_path
                })
            
            # Large class smell
            if self._is_large_class(structure):
                patterns["code_smells"].append({
                    "smell": "large_class",
                    "file": file_path,
                    "details": f"{len(structure.get('classes', []))} classes"
                })
        
        return patterns
    
    def _is_singleton(self, structure: Dict) -> bool:
        """Check if structure suggests singleton"""
        classes = structure.get("classes", [])
        
        for cls in classes:
            methods = cls.get("methods", [])
            if "getInstance" in methods or "__new__" in methods:
                return True
        
        return False
    
    def _is_large_class(self, structure: Dict) -> bool:
        """Check if class is too large"""
        classes = structure.get("classes", [])
        
        for cls in classes:
            if len(cls.get("methods", [])) > 20:
                return True
        
        return False

class DocumentationGenerator:
    """Generate project documentation"""
    
    async def generate(
        self,
        analysis: Dict[str, Any],
        format: str = "markdown"
    ) -> str:
        """Generate documentation"""
        if format == "markdown":
            return self._generate_markdown(analysis)
        else:
            return "Unsupported format"
    
    def _generate_markdown(self, analysis: Dict[str, Any]) -> str:
        """Generate markdown documentation"""
        sections = []
        
        # Title
        sections.append("# Project Analysis Report\n")
        
        # Overview
        sections.append("## Overview\n")
        if "insights" in analysis:
            sections.append(analysis["insights"])
        
        # Structure
        sections.append("\n## Project Structure\n")
        if "structure" in analysis:
            structure = analysis["structure"]
            sections.append(f"- Total files: {len(analysis.get('files', []))}")
            sections.append(f"- Directories: {structure.get('directory_count', 0)}")
            sections.append(f"- Max depth: {structure.get('max_depth', 0)}")
        
        # Languages
        sections.append("\n## Languages\n")
        if "metadata" in analysis:
            languages = {}
            for meta in analysis["metadata"].values():
                lang = meta.get("language", "unknown")
                languages[lang] = languages.get(lang, 0) + 1
            
            for lang, count in sorted(languages.items()):
                sections.append(f"- {lang}: {count} files")
        
        # Dependencies
        sections.append("\n## Dependencies\n")
        if "dependencies" in analysis:
            deps = analysis["dependencies"]
            sections.append(f"External dependencies: {len(deps.get('external', []))}")
            
            if deps.get("circular"):
                sections.append("\n### Circular Dependencies")
                for cycle in deps["circular"]:
                    sections.append(f"- {' -> '.join(cycle)}")
        
        # Patterns
        if "patterns" in analysis:
            patterns = analysis["patterns"]
            
            if patterns.get("design_patterns"):
                sections.append("\n## Design Patterns\n")
                for pattern in patterns["design_patterns"]:
                    sections.append(f"- {pattern['pattern']}: {pattern['file']}")
            
            if patterns.get("code_smells"):
                sections.append("\n## Code Smells\n")
                for smell in patterns["code_smells"]:
                    sections.append(f"- {smell['smell']}: {smell['file']}")
        
        return "\n".join(sections)
```

### Task 3: State Machine Implementation (30 min)
Create `src/state/analyzer_states.py`:

```python
from typing import Optional, Dict, Any, List
import asyncio
import structlog

from src.state.core import State, StateMachine, StateContext, StateType
from src.async_patterns.concurrent_tools import ToolCall

logger = structlog.get_logger()

class AnalyzerState(State):
    """Base state for analyzer"""
    
    def __init__(self, name: str, agent=None, **kwargs):
        super().__init__(name, **kwargs)
        self.agent = agent

class QuestionReceivedState(AnalyzerState):
    """Handle incoming question"""
    
    def __init__(self, agent=None):
        super().__init__("question_received", agent, state_type=StateType.INITIAL)
    
    async def execute(self, context: StateContext) -> Optional[str]:
        """Process question"""
        question = context.get("question")
        analysis = context.get("analysis")
        
        if not question:
            context.set("error", "No question provided")
            return "error"
        
        logger.info(f"Processing question: {question}")
        
        # Determine question type
        question_type = await self._classify_question(question)
        context.set("question_type", question_type)
        
        if question_type == "search":
            return "searching"
        elif question_type == "explain":
            return "explaining"
        elif question_type == "analyze":
            return "analyzing"
        else:
            return "general_response"
    
    async def _classify_question(self, question: str) -> str:
        """Classify question type"""
        question_lower = question.lower()
        
        if any(word in question_lower for word in ["find", "search", "where", "locate"]):
            return "search"
        elif any(word in question_lower for word in ["explain", "what is", "how does"]):
            return "explain"
        elif any(word in question_lower for word in ["analyze", "review", "check"]):
            return "analyze"
        else:
            return "general"

class SearchingState(AnalyzerState):
    """Search for information"""
    
    def __init__(self, agent=None):
        super().__init__("searching", agent)
    
    async def execute(self, context: StateContext) -> Optional[str]:
        """Execute search"""
        question = context.get("question")
        analysis = context.get("analysis")
        
        # Extract search terms
        search_terms = self._extract_search_terms(question)
        
        # Search in analysis
        results = await self._search_analysis(search_terms, analysis)
        
        context.set("search_results", results)
        
        if results:
            return "formatting_response"
        else:
            return "no_results"
    
    def _extract_search_terms(self, question: str) -> List[str]:
        """Extract search terms from question"""
        # Simple extraction - remove common words
        stop_words = {"find", "search", "where", "is", "the", "a", "an", "for"}
        words = question.lower().split()
        return [w for w in words if w not in stop_words and len(w) > 2]
    
    async def _search_analysis(
        self,
        terms: List[str],
        analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Search in analysis data"""
        results = []
        
        # Search in files
        if "files" in analysis:
            for file in analysis["files"]:
                score = sum(
                    1 for term in terms
                    if term in file["relative_path"].lower()
                )
                if score > 0:
                    results.append({
                        "type": "file",
                        "path": file["relative_path"],
                        "score": score
                    })
        
        # Search in metadata
        if "metadata" in analysis:
            for file_path, meta in analysis["metadata"].items():
                if "structure" in meta:
                    structure = meta["structure"]
                    
                    # Search in class names
                    for cls in structure.get("classes", []):
                        score = sum(
                            1 for term in terms
                            if term in cls["name"].lower()
                        )
                        if score > 0:
                            results.append({
                                "type": "class",
                                "name": cls["name"],
                                "file": file_path,
                                "score": score
                            })
                    
                    # Search in function names
                    for func in structure.get("functions", []):
                        score = sum(
                            1 for term in terms
                            if term in func["name"].lower()
                        )
                        if score > 0:
                            results.append({
                                "type": "function",
                                "name": func["name"],
                                "file": file_path,
                                "score": score
                            })
        
        # Sort by score
        results.sort(key=lambda x: x["score"], reverse=True)
        
        return results[:10]  # Top 10 results

class ExplainingState(AnalyzerState):
    """Explain code or concepts"""
    
    def __init__(self, agent=None):
        super().__init__("explaining", agent)
    
    async def execute(self, context: StateContext) -> Optional[str]:
        """Generate explanation"""
        question = context.get("question")
        analysis = context.get("analysis")
        
        # Find relevant context
        relevant_context = await self._find_relevant_context(question, analysis)
        
        # Generate explanation using LLM
        if self.agent and self.agent.llm:
            prompt = self._build_explanation_prompt(question, relevant_context)
            explanation = await self.agent.llm.complete(prompt)
            
            context.set("explanation", explanation)
            return "formatting_response"
        else:
            return "error"
    
    async def _find_relevant_context(
        self,
        question: str,
        analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Find context relevant to question"""
        # Extract what needs explaining
        subjects = self._extract_subjects(question)
        
        context = {}
        
        # Find relevant files
        if "metadata" in analysis:
            for file_path, meta in analysis["metadata"].items():
                for subject in subjects:
                    if subject in file_path.lower():
                        context[file_path] = meta
                        break
        
        return context
    
    def _extract_subjects(self, question: str) -> List[str]:
        """Extract subjects from question"""
        # Simple extraction
        words = question.lower().split()
        subjects = []
        
        for i, word in enumerate(words):
            if word in ["explain", "what", "how"] and i + 1 < len(words):
                subjects.append(words[i + 1])
        
        return subjects
    
    def _build_explanation_prompt(
        self,
        question: str,
        context: Dict[str, Any]
    ) -> str:
        """Build prompt for explanation"""
        prompt = f"Question: {question}\n\n"
        
        if context:
            prompt += "Relevant code context:\n"
            for file_path, meta in context.items():
                prompt += f"\nFile: {file_path}\n"
                if "structure" in meta:
                    prompt += f"Structure: {meta['structure']}\n"
        
        prompt += "\nProvide a clear explanation."
        
        return prompt

class AnalyzingState(AnalyzerState):
    """Perform analysis"""
    
    def __init__(self, agent=None):
        super().__init__("analyzing", agent)
    
    async def execute(self, context: StateContext) -> Optional[str]:
        """Execute analysis"""
        question = context.get("question")
        analysis = context.get("analysis")
        
        # Determine analysis type
        analysis_type = self._determine_analysis_type(question)
        
        # Perform analysis
        if analysis_type == "quality":
            result = await self._analyze_quality(analysis)
        elif analysis_type == "security":
            result = await self._analyze_security(analysis)
        elif analysis_type == "performance":
            result = await self._analyze_performance(analysis)
        else:
            result = await self._general_analysis(question, analysis)
        
        context.set("analysis_result", result)
        return "formatting_response"
    
    def _determine_analysis_type(self, question: str) -> str:
        """Determine type of analysis needed"""
        question_lower = question.lower()
        
        if any(word in question_lower for word in ["quality", "clean", "maintainable"]):
            return "quality"
        elif any(word in question_lower for word in ["security", "vulnerable", "safe"]):
            return "security"
        elif any(word in question_lower for word in ["performance", "speed", "optimize"]):
            return "performance"
        else:
            return "general"
    
    async def _analyze_quality(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze code quality"""
        quality_metrics = {
            "issues": [],
            "score": 100
        }
        
        # Check for code smells
        if "patterns" in analysis and "code_smells" in analysis["patterns"]:
            for smell in analysis["patterns"]["code_smells"]:
                quality_metrics["issues"].append({
                    "type": "code_smell",
                    "description": smell["smell"],
                    "file": smell["file"],
                    "severity": "medium"
                })
                quality_metrics["score"] -= 5
        
        # Check for circular dependencies
        if "dependencies" in analysis and analysis["dependencies"].get("circular"):
            for cycle in analysis["dependencies"]["circular"]:
                quality_metrics["issues"].append({
                    "type": "circular_dependency",
                    "description": f"Circular dependency: {' -> '.join(cycle)}",
                    "severity": "high"
                })
                quality_metrics["score"] -= 10
        
        return quality_metrics
    
    async def _analyze_security(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze security issues"""
        security_issues = []
        
        # Check for hardcoded secrets
        if "patterns" in analysis:
            # Would check pattern results for security issues
            pass
        
        return {"security_issues": security_issues}
    
    async def _analyze_performance(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze performance issues"""
        performance_issues = []
        
        # Check for large files
        if "files" in analysis:
            for file in analysis["files"]:
                if file["size"] > 100000:  # 100KB
                    performance_issues.append({
                        "type": "large_file",
                        "file": file["relative_path"],
                        "size": file["size"]
                    })
        
        return {"performance_issues": performance_issues}
    
    async def _general_analysis(
        self,
        question: str,
        analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """General analysis using LLM"""
        if self.agent and self.agent.llm:
            prompt = f"""
Question: {question}

Project analysis summary:
{self._summarize_analysis(analysis)}

Provide analysis and insights.
"""
            
            result = await self.agent.llm.complete(prompt)
            return {"analysis": result}
        
        return {"error": "Unable to perform analysis"}
    
    def _summarize_analysis(self, analysis: Dict[str, Any]) -> str:
        """Summarize analysis for LLM"""
        summary = []
        
        if "files" in analysis:
            summary.append(f"Total files: {len(analysis['files'])}")
        
        if "structure" in analysis:
            structure = analysis["structure"]
            summary.append(f"Directories: {structure.get('directory_count', 0)}")
        
        if "dependencies" in analysis:
            deps = analysis["dependencies"]
            summary.append(f"External dependencies: {len(deps.get('external', []))}")
        
        return "\n".join(summary)

class FormattingResponseState(AnalyzerState):
    """Format the final response"""
    
    def __init__(self, agent=None):
        super().__init__("formatting_response", agent)
    
    async def execute(self, context: StateContext) -> Optional[str]:
        """Format response"""
        question_type = context.get("question_type")
        
        if question_type == "search":
            response = self._format_search_results(context)
        elif question_type == "explain":
            response = self._format_explanation(context)
        elif question_type == "analyze":
            response = self._format_analysis(context)
        else:
            response = self._format_general(context)
        
        context.set("answer", response)
        return "complete"
    
    def _format_search_results(self, context: StateContext) -> str:
        """Format search results"""
        results = context.get("search_results", [])
        
        if not results:
            return "No results found for your search."
        
        lines = ["Here are the search results:\n"]
        
        for i, result in enumerate(results[:5], 1):
            if result["type"] == "file":
                lines.append(f"{i}. File: {result['path']}")
            elif result["type"] == "class":
                lines.append(f"{i}. Class '{result['name']}' in {result['file']}")
            elif result["type"] == "function":
                lines.append(f"{i}. Function '{result['name']}' in {result['file']}")
        
        return "\n".join(lines)
    
    def _format_explanation(self, context: StateContext) -> str:
        """Format explanation"""
        explanation = context.get("explanation", "Unable to generate explanation.")
        return explanation
    
    def _format_analysis(self, context: StateContext) -> str:
        """Format analysis results"""
        result = context.get("analysis_result", {})
        
        if "issues" in result:
            # Quality analysis
            lines = [f"Code Quality Score: {result.get('score', 'N/A')}/100\n"]
            
            if result["issues"]:
                lines.append("Issues found:")
                for issue in result["issues"]:
                    lines.append(f"- {issue['description']} ({issue['severity']})")
            else:
                lines.append("No issues found!")
            
            return "\n".join(lines)
        
        elif "analysis" in result:
            return result["analysis"]
        
        return "Analysis complete."
    
    def _format_general(self, context: StateContext) -> str:
        """Format general response"""
        return "I've processed your request."

class NoResultsState(AnalyzerState):
    """Handle no results case"""
    
    def __init__(self, agent=None):
        super().__init__("no_results", agent)
    
    async def execute(self, context: StateContext) -> Optional[str]:
        """Handle no results"""
        question = context.get("question")
        
        response = f"I couldn't find any results for: {question}\n"
        response += "Try rephrasing your question or being more specific."
        
        context.set("answer", response)
        return "complete"

class CompleteState(AnalyzerState):
    """Final state"""
    
    def __init__(self, agent=None):
        super().__init__("complete", agent, state_type=StateType.FINAL)
    
    async def execute(self, context: StateContext) -> Optional[str]:
        """Complete processing"""
        logger.info("Question processing complete")
        return None

class AnalyzerStateMachine:
    """State machine for analyzer agent"""
    
    def __init__(self, agent, config):
        self.agent = agent
        self.config = config
        self.machine = StateMachine("analyzer")
        self._setup_states()
    
    def _setup_states(self):
        """Setup all states"""
        states = [
            QuestionReceivedState(self.agent),
            SearchingState(self.agent),
            ExplainingState(self.agent),
            AnalyzingState(self.agent),
            FormattingResponseState(self.agent),
            NoResultsState(self.agent),
            CompleteState(self.agent),
        ]
        
        for state in states:
            self.machine.add_state(
                state,
                initial=(state.name == "question_received"),
                final=(state.name == "complete")
            )
    
    async def initialize(self):
        """Initialize state machine"""
        # Could add persistence, monitoring, etc.
        pass
    
    async def run(
        self,
        initial_state: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run state machine"""
        state_context = StateContext()
        state_context.data = context
        
        # Set initial state
        self.machine.current_state = initial_state
        
        # Run machine
        result = await self.machine.start(state_context)
        
        return result.data
    
    @property
    def current_state(self) -> Optional[str]:
        """Get current state"""
        return self.machine.current_state
```

## 🌞 Afternoon Tasks (90-120 minutes)

### Task 4: CLI Interface (45 min)
Create `src/cli/analyzer_cli.py`:

```python
import asyncio
import click
import json
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax
from rich.panel import Panel
from rich.markdown import Markdown
import structlog

from src.config.settings import AnalyzerConfig
from src.core.analyzer_agent import FileAnalyzerAgent

# Configure logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.dev.ConsoleRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

console = Console()

@click.group()
@click.option('--project', '-p', default='.', help='Project directory to analyze')
@click.option('--config', '-c', help='Config file path')
@click.pass_context
def cli(ctx, project, config):
    """File Analyzer Agent - AI-powered code analysis"""
    ctx.ensure_object(dict)
    
    # Load configuration
    if config:
        ctx.obj['config'] = AnalyzerConfig.from_yaml(config)
    else:
        ctx.obj['config'] = AnalyzerConfig.from_env(project)
    
    # Create agent
    ctx.obj['agent'] = FileAnalyzerAgent(ctx.obj['config'])

@cli.command()
@click.pass_context
def analyze(ctx):
    """Analyze the entire project"""
    agent = ctx.obj['agent']
    
    async def run_analysis():
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Initializing agent...", total=None)
            await agent.initialize()
            
            progress.update(task, description="Analyzing project...")
            result = await agent.analyze_project()
            
            progress.update(task, description="Complete!")
        
        return result
    
    with console.status("[bold green]Analyzing project..."):
        result = asyncio.run(run_analysis())
    
    # Display results
    console.print("\n[bold green]✓ Analysis Complete![/bold green]\n")
    
    # Show summary
    if "insights" in result:
        console.print(Panel(
            result["insights"],
            title="[bold]Project Insights[/bold]",
            border_style="green"
        ))
    
    # Show statistics
    if "files" in result:
        table = Table(title="Project Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        
        table.add_row("Total Files", str(len(result["files"])))
        
        if "structure" in result:
            structure = result["structure"]
            table.add_row("Directories", str(structure.get("directory_count", 0)))
            table.add_row("Max Depth", str(structure.get("max_depth", 0)))
        
        if "dependencies" in result:
            deps = result["dependencies"]
            table.add_row("External Dependencies", str(len(deps.get("external", []))))
            table.add_row("Circular Dependencies", str(len(deps.get("circular", []))))
        
        console.print(table)
    
    # Save results
    output_file = ctx.obj['config'].cache_dir / "analysis_result.json"
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    console.print(f"\n[dim]Results saved to: {output_file}[/dim]")

@cli.command()
@click.argument('question')
@click.pass_context
def ask(ctx, question):
    """Ask a question about the codebase"""
    agent = ctx.obj['agent']
    
    async def get_answer():
        await agent.initialize()
        return await agent.answer_question(question)
    
    with console.status("[bold green]Thinking..."):
        answer = asyncio.run(get_answer())
    
    console.print("\n[bold]Answer:[/bold]")
    console.print(Panel(answer, border_style="blue"))

@cli.command()
@click.argument('pattern')
@click.pass_context
def find(ctx, pattern):
    """Find a pattern in the codebase"""
    agent = ctx.obj['agent']
    
    async def search():
        await agent.initialize()
        return await agent.find_pattern(pattern)
    
    with console.status(f"[bold green]Searching for '{pattern}'..."):
        results = asyncio.run(search())
    
    if not results:
        console.print(f"\n[yellow]No matches found for '{pattern}'[/yellow]")
        return
    
    console.print(f"\n[bold green]Found {len(results)} matches:[/bold green]\n")
    
    # Group by file
    by_file = {}
    for result in results:
        file = result["file"]
        if file not in by_file:
            by_file[file] = []
        by_file[file].append(result)
    
    # Display results
    for file, matches in by_file.items():
        console.print(f"[bold cyan]{file}[/bold cyan]")
        for match in matches:
            console.print(f"  Line {match['line']}: [yellow]{match['match']}[/yellow]")
        console.print()

@cli.command()
@click.option('--format', '-f', default='markdown', help='Output format (markdown, html)')
@click.option('--output', '-o', help='Output file')
@click.pass_context
def docs(ctx, format, output):
    """Generate project documentation"""
    agent = ctx.obj['agent']
    
    async def generate():
        await agent.initialize()
        return await agent.generate_documentation(output_format=format)
    
    with console.status("[bold green]Generating documentation..."):
        documentation = asyncio.run(generate())
    
    if output:
        with open(output, 'w') as f:
            f.write(documentation)
        console.print(f"\n[bold green]✓ Documentation saved to: {output}[/bold green]")
    else:
        console.print("\n[bold]Generated Documentation:[/bold]\n")
        if format == 'markdown':
            console.print(Markdown(documentation))
        else:
            console.print(documentation)

@cli.command()
@click.pass_context
def status(ctx):
    """Show agent status"""
    agent = ctx.obj['agent']
    
    async def get_status():
        await agent.initialize()
        return agent.get_status()
    
    status = asyncio.run(get_status())
    
    table = Table(title="Agent Status")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="magenta")
    
    table.add_row("Initialized", "✓" if status["initialized"] else "✗")
    table.add_row("Cached Analyses", str(len(status["cached_analyses"])))
    table.add_row("Current State", status.get("state_machine_state", "N/A"))
    table.add_row("Cost Today", f"${status.get('cost_today', 0):.4f}")
    
    console.print(table)

@cli.command()
@click.pass_context
def interactive(ctx):
    """Start interactive mode"""
    agent = ctx.obj['agent']
    
    async def initialize():
        await agent.initialize()
    
    console.print("[bold green]File Analyzer Agent - Interactive Mode[/bold green]")
    console.print("Type 'help' for commands, 'exit' to quit\n")
    
    # Initialize agent
    asyncio.run(initialize())
    
    while True:
        try:
            command = console.input("[bold blue]> [/bold blue]")
            
            if command.lower() in ['exit', 'quit']:
                break
            elif command.lower() == 'help':
                show_help()
            elif command.lower().startswith('find '):
                pattern = command[5:]
                results = asyncio.run(agent.find_pattern(pattern))
                display_search_results(results, pattern)
            elif command.lower() == 'analyze':
                with console.status("[bold green]Analyzing..."):
                    result = asyncio.run(agent.analyze_project())
                console.print("[bold green]✓ Analysis complete![/bold green]")
            elif command.lower() == 'status':
                status = agent.get_status()
                display_status(status)
            else:
                # Treat as question
                with console.status("[bold green]Thinking..."):
                    answer = asyncio.run(agent.answer_question(command))
                console.print(f"\n[bold]Answer:[/bold] {answer}\n")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
    
    console.print("\n[bold green]Goodbye![/bold green]")

def show_help():
    """Show help information"""
    help_text = """
[bold]Available Commands:[/bold]
  help              Show this help message
  analyze           Analyze the entire project
  find <pattern>    Search for a pattern in the codebase
  status            Show agent status
  exit/quit         Exit interactive mode
  
[bold]Or just type a question about your codebase![/bold]
"""
    console.print(Panel(help_text, title="Help", border_style="blue"))

def display_search_results(results, pattern):
    """Display search results"""
    if not results:
        console.print(f"[yellow]No matches found for '{pattern}'[/yellow]")
        return
    
    console.print(f"\n[bold green]Found {len(results)} matches:[/bold green]")
    
    for i, result in enumerate(results[:10], 1):
        console.print(f"{i}. [cyan]{result['file']}[/cyan]:{result['line']}")
        console.print(f"   [yellow]{result['match']}[/yellow]")
    
    if len(results) > 10:
        console.print(f"\n[dim]... and {len(results) - 10} more[/dim]")

def display_status(status):
    """Display agent status"""
    console.print("\n[bold]Agent Status:[/bold]")
    for key, value in status.items():
        console.print(f"  {key}: [cyan]{value}[/cyan]")

if __name__ == '__main__':
    cli()
```

Create `main.py`:

```python
#!/usr/bin/env python3
"""
File Analyzer Agent - AI-powered code analysis tool
"""

from src.cli.analyzer_cli import cli

if __name__ == '__main__':
    cli()
```

### Task 5: Testing Suite (30 min)
Create `tests/test_analyzer.py`:

```python
import pytest
import asyncio
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch

from src.config.settings import AnalyzerConfig
from src.core.analyzer_agent import FileAnalyzerAgent
from src.tools.code_analysis_tools import FileScanner, CodeParser

class TestFileAnalyzerAgent:
    @pytest.fixture
    def config(self, tmp_path):
        """Create test configuration"""
        return AnalyzerConfig(
            project_root=tmp_path,
            cache_dir=tmp_path / ".cache",
            log_dir=tmp_path / "logs",
            max_files_to_analyze=10
        )
    
    @pytest.fixture
    def agent(self, config):
        """Create test agent"""
        with patch('src.core.analyzer_agent.create_llm_client'):
            return FileAnalyzerAgent(config)
    
    @pytest.mark.asyncio
    async def test_initialization(self, agent):
        """Test agent initialization"""
        await agent.initialize()
        assert agent.initialized
    
    @pytest.mark.asyncio
    async def test_analyze_project(self, agent, tmp_path):
        """Test project analysis"""
        # Create test files
        (tmp_path / "test.py").write_text("def hello(): pass")
        (tmp_path / "main.py").write_text("import test\ntest.hello()")
        
        # Mock LLM response
        agent.llm = AsyncMock()
        agent.llm.complete = AsyncMock(return_value="Test insights")
        
        await agent.initialize()
        result = await agent.analyze_project()
        
        assert "files" in result
        assert len(result["files"]) == 2
        assert "insights" in result
    
    @pytest.mark.asyncio
    async def test_answer_question(self, agent):
        """Test question answering"""
        # Mock components
        agent.llm = AsyncMock()
        agent.llm.complete = AsyncMock(return_value="Test answer")
        
        await agent.initialize()
        
        # Mock analysis cache
        agent.analysis_cache["full_analysis"] = {
            "files": [{"relative_path": "test.py"}],
            "metadata": {}
        }
        
        answer = await agent.answer_question("What files are in the project?")
        assert answer == "Here are the search results:\n\n1. File: test.py"
    
    @pytest.mark.asyncio
    async def test_find_pattern(self, agent, tmp_path):
        """Test pattern finding"""
        # Create test file
        test_file = tmp_path / "test.py"
        test_file.write_text("# TODO: Fix this\ndef broken(): pass")
        
        await agent.initialize()
        
        # Mock file list
        agent.analysis_cache["structure"] = {
            "files": [{
                "path": str(test_file),
                "relative_path": "test.py"
            }]
        }
        
        results = await agent.find_pattern("TODO")
        assert len(results) > 0

class TestCodeAnalysisTools:
    @pytest.mark.asyncio
    async def test_file_scanner(self, tmp_path):
        """Test file scanning"""
        # Create test structure
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.py").write_text("print('hello')")
        (tmp_path / "src" / "test.js").write_text("console.log('test')")
        (tmp_path / "README.md").write_text("# Test")
        
        config = AnalyzerConfig(project_root=tmp_path)
        scanner = FileScanner(config)
        
        result = await scanner.execute({"directory": str(tmp_path)})
        
        assert result.success
        files = result.result
        assert len(files) == 2  # Only .py and .js files
    
    @pytest.mark.asyncio
    async def test_code_parser_python(self):
        """Test Python code parsing"""
        parser = CodeParser()
        
        code = """
import os
from pathlib import Path

class TestClass:
    def method1(self):
        pass
    
    def method2(self, arg):
        return arg

def standalone_function():
    return 42
"""
        
        result = await parser.parse_python(code)
        
        assert len(result["imports"]) == 2
        assert len(result["classes"]) == 1
        assert result["classes"][0]["name"] == "TestClass"
        assert len(result["classes"][0]["methods"]) == 2
        assert len(result["functions"]) == 1
        assert result["functions"][0]["name"] == "standalone_function"
    
    @pytest.mark.asyncio
    async def test_pattern_finder(self, tmp_path):
        """Test pattern finding"""
        from src.tools.code_analysis_tools import PatternFinder
        
        # Create test file
        test_file = tmp_path / "test.py"
        test_file.write_text("""
# TODO: Implement this
# FIXME: Bug here
password = "secret123"  # Security issue
print("debug")  # Debug statement
try:
    something()
except:  # Catch-all exception
    pass
""")
        
        finder = PatternFinder()
        
        files = [{
            "path": str(test_file),
            "relative_path": "test.py"
        }]
        
        result = await finder.execute({"files": files})
        
        assert result.success
        patterns = result.result
        assert "todo" in patterns
        assert "fixme" in patterns
        assert "security" in patterns

class TestStateMachine:
    @pytest.mark.asyncio
    async def test_question_flow(self):
        """Test question answering state flow"""
        from src.state.analyzer_states import AnalyzerStateMachine
        
        # Mock agent
        agent = Mock()
        agent.llm = AsyncMock()
        
        config = Mock()
        
        sm = AnalyzerStateMachine(agent, config)
        await sm.initialize()
        
        # Test search question
        context = {
            "question": "Where is the main function?",
            "analysis": {
                "files": [{"relative_path": "main.py"}],
                "metadata": {
                    "main.py": {
                        "structure": {
                            "functions": [{"name": "main", "line": 10}]
                        }
                    }
                }
            }
        }
        
        result = await sm.run("question_received", context)
        
        assert "answer" in result
        assert "main" in result["answer"]

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

### Task 6: Documentation and Examples (15 min)
Create `README.md`:

```markdown
# File Analyzer Agent

AI-powered code analysis tool that understands your codebase and answers questions about it.

## Features

- 🔍 **Intelligent Code Search**: Find functions, classes, and patterns across your codebase
- 📊 **Project Analysis**: Understand project structure, dependencies, and complexity
- 💬 **Natural Language Q&A**: Ask questions about your code in plain English
- 📝 **Documentation Generation**: Automatically generate project documentation
- 🔒 **Security Analysis**: Identify potential security issues and vulnerabilities
- 📈 **Code Quality Metrics**: Assess code quality and identify improvements

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd file_analyzer_agent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Configuration

Set up your environment variables:

```bash
# Create .env file
cp .env.example .env

# Edit .env with your API keys
OPENAI_API_KEY=your-api-key-here
# or
ANTHROPIC_API_KEY=your-api-key-here
LLM_PROVIDER=anthropic
```

## Usage

### Command Line Interface

```bash
# Analyze a project
python main.py --project /path/to/project analyze

# Ask a question
python main.py ask "What does the main function do?"

# Find patterns
python main.py find "TODO"

# Generate documentation
python main.py docs --output project-docs.md

# Interactive mode
python main.py interactive
```

### Python API

```python
from src.config.settings import AnalyzerConfig
from src.core.analyzer_agent import FileAnalyzerAgent

# Initialize agent
config = AnalyzerConfig(project_root="./my_project")
agent = FileAnalyzerAgent(config)
await agent.initialize()

# Analyze project
analysis = await agent.analyze_project()

# Ask questions
answer = await agent.answer_question("What are the main components?")

# Find patterns
results = await agent.find_pattern("security")
```

## Examples

### Analyzing a Python Project

```bash
$ python main.py --project ~/code/my_app analyze

✓ Analysis Complete!

╭─────────────────── Project Insights ──────────────────────╮
│ This is a Flask web application with a RESTful API.       │
│ The project follows MVC architecture with clear           │
│ separation of concerns. Main components include:          │
│ - API endpoints in routes/                                │
│ - Data models in models/                                  │
│ - Business logic in services/                             │
│                                                           │
│ Suggestions for improvement:                              │
│ - Add more comprehensive error handling                   │
│ - Implement input validation for API endpoints            │
│ - Consider adding caching for database queries            │
╰──────────────────────────────────────────────────────────╯

Project Statistics
┏━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┓
┃ Metric               ┃ Value  ┃
┡━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━┩
│ Total Files          │ 47     │
│ Directories          │ 12     │
│ Max Depth            │ 4      │
│ External Dependencies│ 15     │
│ Circular Dependencies│ 0      │
└──────────────────────┴────────┘
```

### Interactive Q&A

```bash
$ python main.py interactive

File Analyzer Agent - Interactive Mode
Type 'help' for commands, 'exit' to quit

> How is authentication implemented?

Answer: Authentication is implemented using JWT tokens. The auth 
module in services/auth.py handles token generation and validation. 
User credentials are verified against the database, and successful 
authentication returns a JWT token that must be included in the 
Authorization header for protected endpoints.

> find security issues

Found 3 matches:
1. config.py:12
   password = "admin123"  # FIXME: hardcoded password
2. api/users.py:45
   # TODO: Add rate limiting to prevent brute force
3. models/user.py:23
   # Security: passwords should be hashed

> What files handle database operations?

Answer: Database operations are primarily handled by:
1. models/__init__.py - Database initialization
2. models/user.py - User model and queries
3. models/product.py - Product model and queries
4. services/database.py - Database connection management
5. migrations/ - Database migration scripts
```

## Architecture

The File Analyzer Agent uses a modular architecture:

- **Core Agent**: Orchestrates analysis and question answering
- **State Machine**: Manages conversation flow and context
- **Tool System**: Extensible tools for code analysis
- **LLM Integration**: Handles AI model interactions
- **Async Patterns**: Ensures high performance

## Extending

Add new analysis tools by implementing the `Tool` interface:

```python
from src.tools.base import Tool, ToolResult

class MyCustomTool(Tool):
    def __init__(self):
        super().__init__(
            name="my_tool",
            description="My custom analysis tool"
        )
    
    async def execute(self, input_data: Dict) -> ToolResult:
        # Your analysis logic here
        return ToolResult(
            success=True,
            output="Analysis complete",
            result={"data": "..."}
        )
```

## Performance

- Concurrent file processing for speed
- Intelligent caching to reduce API calls
- Cost tracking and limits
- Async execution throughout

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

MIT License - see LICENSE file for details
```

Create `requirements.txt`:

```
# Core dependencies
openai==1.12.0
anthropic==0.18.1
python-dotenv==1.0.0
pydantic==2.5.0
structlog==24.1.0

# Async and networking
aiohttp==3.9.0
aiofiles==23.2.1
tenacity==8.2.3

# CLI and display
click==8.1.7
rich==13.7.0

# Parsing and analysis
pyyaml==6.0.1
pyparsing==3.1.1
json5==0.9.14
hjson==3.1.0
xmltodict==0.13.0

# Code parsing
ast-grep-py==0.12.0
tree-sitter==0.20.4

# Natural language processing (optional)
spacy==3.7.2
tiktoken==0.5.2

# Testing
pytest==8.0.0
pytest-asyncio==0.23.0
pytest-cov==4.1.0

# Development
black==24.1.0
flake8==7.0.0
mypy==1.8.0
```

## 🌆 Evening Tasks (45-60 minutes)

### Task 7: Integration and Demo (30 min)
Create `demo.py`:

```python
#!/usr/bin/env python3
"""
Demo script for File Analyzer Agent
"""

import asyncio
import sys
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.progress import track

from src.config.settings import AnalyzerConfig
from src.core.analyzer_agent import FileAnalyzerAgent

console = Console()

async def run_demo():
    """Run demonstration of agent capabilities"""
    console.print(Panel.fit(
        "[bold green]File Analyzer Agent Demo[/bold green]\n"
        "AI-powered code analysis and Q&A",
        border_style="green"
    ))
    
    # Use current directory as demo project
    project_path = Path.cwd()
    console.print(f"\n[cyan]Analyzing project:[/cyan] {project_path}\n")
    
    # Initialize agent
    config = AnalyzerConfig(project_root=project_path)
    agent = FileAnalyzerAgent(config)
    
    with console.status("[bold green]Initializing agent..."):
        await agent.initialize()
    
    # Demo 1: Project Analysis
    console.print("\n[bold]Demo 1: Project Analysis[/bold]")
    console.print("-" * 40)
    
    with console.status("[bold green]Analyzing project structure..."):
        analysis = await agent.analyze_project()
    
    console.print("[green]✓[/green] Analysis complete!")
    
    if "insights" in analysis:
        console.print("\n[bold]Insights:[/bold]")
        console.print(Panel(analysis["insights"], border_style="blue"))
    
    # Demo 2: Q&A
    console.print("\n[bold]Demo 2: Question Answering[/bold]")
    console.print("-" * 40)
    
    questions = [
        "What is the main purpose of this project?",
        "What are the key files I should look at?",
        "Are there any TODO items or FIXME comments?",
        "What external dependencies does this project use?"
    ]
    
    for question in questions:
        console.print(f"\n[yellow]Q:[/yellow] {question}")
        
        with console.status("[bold green]Thinking..."):
            answer = await agent.answer_question(question)
        
        console.print(f"[green]A:[/green] {answer}")
        await asyncio.sleep(1)  # Pause for readability
    
    # Demo 3: Pattern Search
    console.print("\n[bold]Demo 3: Pattern Search[/bold]")
    console.print("-" * 40)
    
    patterns = ["TODO", "FIXME", "import"]
    
    for pattern in patterns:
        console.print(f"\n[cyan]Searching for:[/cyan] {pattern}")
        
        results = await agent.find_pattern(pattern)
        
        if results:
            console.print(f"[green]Found {len(results)} matches[/green]")
            
            # Show first 3 results
            for result in results[:3]:
                console.print(
                    f"  {result['file']}:{result['line']} - "
                    f"[yellow]{result['match']}[/yellow]"
                )
            
            if len(results) > 3:
                console.print(f"  [dim]... and {len(results) - 3} more[/dim]")
        else:
            console.print("[yellow]No matches found[/yellow]")
    
    # Demo 4: Cost Summary
    console.print("\n[bold]Demo 4: Cost Tracking[/bold]")
    console.print("-" * 40)
    
    cost_summary = agent.cost_tracker.get_summary(period_days=1)
    
    console.print(f"Total API calls: {cost_summary.get('count', 0)}")
    console.print(f"Total tokens used: {cost_summary.get('total_tokens', 0)}")
    console.print(f"Estimated cost: ${cost_summary.get('total_cost', 0):.4f}")
    
    console.print("\n[bold green]Demo Complete![/bold green]")
    console.print("\nTry it yourself:")
    console.print("  python main.py interactive")
    console.print("  python main.py ask \"Your question here\"")
    console.print("  python main.py docs")

if __name__ == "__main__":
    try:
        asyncio.run(run_demo())
    except KeyboardInterrupt:
        console.print("\n[yellow]Demo interrupted[/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        sys.exit(1)
```

### Task 8: Performance Testing (15 min)
Create `benchmarks/benchmark_analyzer.py`:

```python
import asyncio
import time
from pathlib import Path
import statistics
from typing import List

from src.config.settings import AnalyzerConfig
from src.core.analyzer_agent import FileAnalyzerAgent

class AnalyzerBenchmark:
    """Benchmark analyzer performance"""
    
    def __init__(self, project_path: Path):
        self.project_path = project_path
        self.results = {}
    
    async def run_benchmarks(self):
        """Run all benchmarks"""
        print("Running File Analyzer Benchmarks...")
        print("=" * 50)
        
        # Initialize agent
        config = AnalyzerConfig(
            project_root=self.project_path,
            enable_caching=False  # Disable caching for benchmarks
        )
        agent = FileAnalyzerAgent(config)
        await agent.initialize()
        
        # Benchmark 1: Project Analysis
        print("\nBenchmark 1: Full Project Analysis")
        analysis_times = []
        
        for i in range(3):
            start = time.time()
            await agent.analyze_project()
            duration = time.time() - start
            analysis_times.append(duration)
            print(f"  Run {i+1}: {duration:.2f}s")
        
        self.results["project_analysis"] = {
            "mean": statistics.mean(analysis_times),
            "min": min(analysis_times),
            "max": max(analysis_times)
        }
        
        # Benchmark 2: Question Answering
        print("\nBenchmark 2: Question Answering")
        questions = [
            "What is the main function?",
            "List all classes in the project",
            "What dependencies are used?",
            "Find all TODO comments",
            "Explain the project structure"
        ]
        
        qa_times = []
        
        for question in questions:
            start = time.time()
            await agent.answer_question(question)
            duration = time.time() - start
            qa_times.append(duration)
            print(f"  '{question[:30]}...': {duration:.2f}s")
        
        self.results["question_answering"] = {
            "mean": statistics.mean(qa_times),
            "min": min(qa_times),
            "max": max(qa_times),
            "total": sum(qa_times)
        }
        
        # Benchmark 3: Pattern Search
        print("\nBenchmark 3: Pattern Search")
        patterns = ["TODO", "import", "class", "def", "error"]
        
        search_times = []
        
        for pattern in patterns:
            start = time.time()
            await agent.find_pattern(pattern)
            duration = time.time() - start
            search_times.append(duration)
            print(f"  Pattern '{pattern}': {duration:.2f}s")
        
        self.results["pattern_search"] = {
            "mean": statistics.mean(search_times),
            "min": min(search_times),
            "max": max(search_times)
        }
        
        # Print summary
        self.print_summary()
    
    def print_summary(self):
        """Print benchmark summary"""
        print("\n" + "=" * 50)
        print("BENCHMARK SUMMARY")
        print("=" * 50)
        
        for benchmark, stats in self.results.items():
            print(f"\n{benchmark.replace('_', ' ').title()}:")
            print(f"  Average: {stats['mean']:.2f}s")
            print(f"  Min: {stats['min']:.2f}s")
            print(f"  Max: {stats['max']:.2f}s")
            
            if "total" in stats:
                print(f"  Total: {stats['total']:.2f}s")
        
        # Calculate throughput
        if "project_analysis" in self.results:
            # Assume project has ~100 files
            throughput = 100 / self.results["project_analysis"]["mean"]
            print(f"\nEstimated throughput: {throughput:.1f} files/second")

async def main():
    """Run benchmarks"""
    # Use current directory or specify a test project
    project_path = Path.cwd()
    
    if len(sys.argv) > 1:
        project_path = Path(sys.argv[1])
    
    benchmark = AnalyzerBenchmark(project_path)
    await benchmark.run_benchmarks()

if __name__ == "__main__":
    import sys
    asyncio.run(main())
```

### Task 9: Final Polish and Documentation

Update your learning journal in CLAUDE.md:

```markdown
## Day 7: Week 1 Project

### What I Built
- ✅ Complete file analyzer agent with all week's learnings
- ✅ Production-ready CLI interface
- ✅ Comprehensive testing and benchmarking

### Project Features
1. **Intelligent Code Analysis**
   - Automatic project structure understanding
   - Dependency analysis and circular detection
   - Pattern recognition and code quality assessment

2. **Natural Language Interface**
   - Answer questions about codebase
   - Explain code functionality
   - Find specific patterns or components

3. **Production Features**
   - Async execution for performance
   - Cost tracking and limits
   - Comprehensive error handling
   - Rich CLI with progress indicators

### Technical Implementation
- **LLM Integration**: Advanced client with streaming and retries
- **Parsing**: Unified parser handling multiple formats
- **State Machine**: Clean conversation flow management
- **Tools**: Extensible tool system for analysis
- **Async**: Concurrent execution throughout

### Performance Metrics
- Analyze 100-file project: ~10 seconds
- Answer question: ~2-3 seconds
- Pattern search: <1 second
- Concurrent file processing: 10x speedup

### Key Learnings
1. **Architecture**: Clean separation of concerns essential
2. **Async**: Massive performance improvements possible
3. **State Management**: Crucial for complex agents
4. **Error Handling**: Must be comprehensive
5. **User Experience**: Rich CLI makes huge difference

### Challenges Overcome
- Complex state management for conversations
- Efficient parsing of various code formats
- Cost optimization while maintaining quality
- Async coordination across components

### Next Week Goals
- Add more sophisticated code understanding
- Implement code generation capabilities
- Build web interface
- Add team collaboration features
```

## 📊 Project Deliverables Checklist
- [ ] Complete file analyzer agent
- [ ] Production CLI interface
- [ ] Code analysis tools
- [ ] State machine for conversations
- [ ] Async execution throughout
- [ ] Cost tracking and limits
- [ ] Comprehensive testing
- [ ] Documentation and examples
- [ ] Performance benchmarks

## 🎯 Success Metrics
Your project succeeds if it can:
1. Analyze a 100+ file codebase in under 30 seconds
2. Answer questions about code accurately
3. Find patterns across multiple files
4. Generate useful documentation
5. Handle errors gracefully
6. Track and limit costs

## 🚀 Extension Ideas
1. **Code Generation**: Add ability to generate code based on descriptions
2. **Refactoring Suggestions**: Provide specific refactoring recommendations
3. **Test Generation**: Automatically generate test cases
4. **CI/CD Integration**: Run as part of build pipeline
5. **Team Features**: Share analyses and insights

## 🎉 Week 1 Complete!

**Congratulations!** You've built a production-ready AI agent from scratch in just one week. You've learned:

- How to structure AI agents properly
- Advanced LLM integration techniques
- Robust parsing strategies
- State machine design
- Async programming patterns
- Production best practices

This file analyzer agent demonstrates all the key concepts and serves as a foundation for more complex agents. Next week, we'll dive into plugin architectures, distributed execution, and advanced agent capabilities.

### Running Your Project

```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Add your API key to .env

# Run the demo
python demo.py

# Try interactive mode
python main.py interactive

# Analyze a project
python main.py --project /path/to/project analyze

# Ask questions
python main.py ask "What does this project do?"

# Generate documentation
python main.py docs --output README_generated.md
```

### Share Your Success!

Don't forget to:
1. Commit your code to GitHub
2. Share your learnings with the community
3. Try analyzing different types of projects
4. Experiment with the tool's capabilities
5. Plan improvements for next week

Ready for Week 2? We'll build on this foundation to create even more sophisticated agent systems!