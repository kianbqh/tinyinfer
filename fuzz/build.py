"""Explicit future build command; never installs a compiler or dependencies."""
import argparse
import shutil
import sys
from pathlib import Path
from metadata import ROOT, environment, run, sha, sources, storage, unique, utc, write


def main():
    p=argparse.ArgumentParser();p.add_argument('--output-parent',required=True);p.add_argument('--cxx',required=True)
    a=p.parse_args();directory=unique(a.output_parent,'build');build=directory/'binary'
    # Keep the clang++ invocation name: resolving its symlink to clang can change
    # the driver's default C++ runtime linkage. Hashing follows the symlink.
    cxx=Path(a.cxx).absolute();cxx.resolve(strict=True);start=sources()
    record={'schema_version':1,'status':'started','start_utc':utc(),'sources_start':start,'environment':environment(),
            'compiler_path':str(cxx),'compiler_sha256':sha(cxx),'commands':[],'build_directory':str(build)}
    write(directory/'build.json',record)
    try:
        commands=[([str(cxx),'--version'],'compiler.log',10),
                  (['cmake','-S',str(ROOT/'fuzz'),'-B',str(build),'-DCMAKE_CXX_COMPILER='+str(cxx),
                    '-DCMAKE_BUILD_TYPE=RelWithDebInfo'],'configure.log',60),
                  (['cmake','--build',str(build),'--parallel','1'],'build.log',300)]
        for argv,name,timeout in commands:
            item=run(argv,directory/name,timeout=timeout);record['commands'].append(item);write(directory/'build.json',record)
            if item['exit_code']!=0:raise RuntimeError(name+' failed or timed out')
        record['binaries']={target:{'path':str(build/('fuzz_'+target)),'sha256':sha(build/('fuzz_'+target))}
                            for target in ('json','artifact','host_validation','cpu_ops','codec')}
        record['compile_commands_sha256']=sha(build/'compile_commands.json');record['sources_end']=sources()
        if record['sources_end']['files']!=start['files']:raise RuntimeError('source changed during build')
        record['status']='built_not_run'
    except Exception as error:record.update(status='failed',error=str(error))
    finally:record['end_utc']=utc();write(directory/'build.json',record)
    print(directory/'build.json')
    return 0 if record['status']=='built_not_run' else 1


if __name__=='__main__':sys.exit(main())
