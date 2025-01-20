# (c) 2025 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

import datetime
import re

C_HDR_FILE: str = '../../magnetron/magnetron.h'
OUTPUT_FILE: str = 'magnetron/_ffi_cdecl_generated.py'

print(f'Generating {OUTPUT_FILE} from {C_HDR_FILE}...')


def comment_replacer(match):
    s = match.group(0)
    if s.startswith('/'):
        return ' '
    else:
        return s


macro_substitutions: dict[str, str] = {
    'MAG_EXPORT': ' ',
    'MAG_MAX_DIMS': str(6),  # SYNC with magnetron.h
    'MAG_MAX_OP_PARAMS': str(6)  # SYNC with magnetron.h
}

def keep_line(line: str) -> bool:
    if line == '' or line.startswith('#'):
        return False
    if line.startswith('extern "C"'):
        return False
    if line.startswith('mag_static_assert'):
        return False
    return True


c_input: list[str] = []
with open(C_HDR_FILE, 'rt') as f:
    full_src: str = f.read()
    pattern = re.compile(
        r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
        re.DOTALL | re.MULTILINE
    )
    full_src = re.sub(pattern, comment_replacer, full_src)  # remove comments
    for macro, replacement in macro_substitutions.items():
        full_src = full_src.replace(macro, replacement)
    c_input = [line.strip() for line in full_src.splitlines()]  # remove empty lines
    c_input = [line for line in c_input if keep_line(line)]  # remove empty lines

out = f'# Autogenered by {__file__} {datetime.datetime.now()}, do NOT edit!\n\n'
out += "__MAG_CDECLS: str = '''\n"
for line in c_input:
    out += f'{line}\n'
out = out.rstrip()
if out.endswith('}'):
    out = out[:-1]
out += "'''\n"

with open(OUTPUT_FILE, 'wt') as f:
    f.write(out)

print(f'Generated {OUTPUT_FILE} with {len(c_input)} lines.')
