"""Record an actual existing server build; does not compile or run the model."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from .metadata import command,digest,metadata,now,write


def main():
    p=argparse.ArgumentParser()
    for key in ("binary","build-directory","build-log","command-record","output"):p.add_argument("--"+key,required=True)
    p.add_argument("--exit-code",type=int,required=True);p.add_argument("--native-library");p.add_argument("--backend-library");a=p.parse_args()
    if a.exit_code:raise ValueError("failed builds retain logs but cannot create accepted binary receipt")
    root=Path(__file__).resolve().parents[1];out=Path(a.output).resolve()
    if out.exists():raise FileExistsError(out)
    binary=Path(a.binary).resolve(strict=True);build=Path(a.build_directory).resolve(strict=True)
    argv=json.loads(Path(a.command_record).read_text(encoding="utf-8"))
    if not isinstance(argv,list) or not argv or any(not isinstance(v,str) for v in argv):raise ValueError("command-record must be exact actual argv JSON array")
    compile_path=build/"compile_commands.json";commands=json.loads(compile_path.read_text(encoding="utf-8"))
    server_commands=[r for r in commands if r["file"].replace("\\","/").endswith("/server/src/main.cpp")]
    if len(server_commands)!=1:raise ValueError("one actual native server compilation command required")
    sources=[p for p in (root/"server").rglob("*") if p.is_file() and p.suffix in (".cpp",".hpp")]
    sources.append(root/"server/CMakeLists.txt")
    sources.append(root/"CMakeLists.txt")
    sources.extend((root/"cmake").glob("*.cmake"))
    libraries={}
    if bool(a.native_library)!=bool(a.backend_library):raise ValueError("both native GPT-2 and backend libraries required together")
    if a.native_library:
        sources.extend([root/"llm/include/tinyinfer_llm.h",root/"llm/src/runtime.cu",root/"llm/src/artifact.hpp"])
        for name,path in (("tinyinfer_llm",a.native_library),("tinyinfer_backend",a.backend_library)):
            file=Path(path).resolve(strict=True);libraries[name]=dict(path=str(file),sha256=digest(file))
    result=dict(timestamp=now(),binary=str(binary),binary_sha256=digest(binary),environment=metadata(root),
                build_command_argv=argv,build_exit_code=0,build_log_sha256=digest(a.build_log),
                compile_commands=server_commands,compile_commands_sha256=digest(compile_path),cmake_cache_sha256=digest(build/"CMakeCache.txt"),
                source_sha256={str(path.relative_to(root)):digest(path) for path in sources},native_libraries=libraries,
                ldd=command(["ldd",str(binary)]))
    write(out,result)

if __name__=="__main__":main()
