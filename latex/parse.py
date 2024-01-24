import argparse
import re



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', default='thesis.tex', help='filename')
    parser.add_argument('--subsec-number', default=None, help='subsection number')
    args = parser.parse_args()
    filename = args.file

    parser_instance = Parser(filename, args.subsec_number)
    for line in parser_instance.filter_lines():
        print(line)

class Parser:
    def __init__(self, filename, n_th_subsection: int | None):
        self.filename = filename
        self.n_th_subsection = n_th_subsection

    def filter_lines(self):
        with open(self.filename, 'r') as file:
            line_number = 1
            env_stack = []
            n_subsecs = 0
            for line_number, line in enumerate(file):
                pattern = r"\\begin{(.*)}"
                match = re.search(pattern, line)
                if match:
                    env = match.group(1)
                    env_stack.append(env)
                pattern = r"\\end{(.*)}"
                match = re.search(pattern, line)
                if match:
                    env_stack.pop()
                
                if (comment_index := line.find('%')) != -1:
                    line = line[:comment_index]
                
                if '\\subsection' in line:
                    n_subsecs += 1

                if self.n_th_subsection is not None:
                    if n_subsecs != int(self.n_th_subsection):
                        continue
                
                if len(line.strip()) == 0 or 'align*' in env_stack or 'figure' in env_stack or 'table' in env_stack or 'itemize' in env_stack or 'align' in env_stack or 'algorithm' in env_stack:
                    continue

                # if in_align_environment or in_figure_environment:
                #     if line.strip().startswith('\\end{align') or line.strip().startswith('\\end{figure'):
                #         in_align_environment = False
                #         in_figure_environment = False
                #     continue
                # if line.strip().startswith('\\begin{align'):
                #     in_align_environment = True
                #     continue
                # if line.strip().startswith('\\begin{figure'):
                #     in_figure_environment = True
                #     continue
                yield f'{line_number}: {line.strip()}'
                # line_number += 1



if __name__ == '__main__':
    main()
