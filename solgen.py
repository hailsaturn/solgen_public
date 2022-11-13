import os, json, csv, subprocess, argparse
from string import Template
import syntaxtree

def parse_json(s):
    try:
        res = json.loads(s)
        if type(res) is dict:
            return res
    except json.JSONDecodeError:
        return False
    return False

class LoadEvalError(Exception):
    def __init__(self, string, message):
        self.string = string
        self.message = message
        super().__init__(message)

    def __str__(self):
        return f'{self.string} -> {self.message}'

marks = {"\\documentclass",
         "#colnames",
         "#csv",
         "#identifier",
         "#evaluation",
         "#writepath",
         }
def is_marker(l):
    for m in marks:
        if l.startswith(m):
            return True
    return False

        
class Importer:
    def __init__(self, filename='template.tex', suffix="solutions", comp=False, digits=None):
        self.suffix = suffix
        self.digits = digits
        self.delimiter = "_"
        self.tex = None
        self.csv = None
        self.evaluation = None
        self.colnames = None
        with open(filename, 'r') as template:
            mode = "config"
            line = template.readline()
            while line != '':
                if line.startswith("\\documentclass"):
                    self.tex = ""
                    while line != '':
                        self.tex += line
                        line = template.readline()
                elif line.startswith("#colnames"):
                    self.colnames = dict()
                    line = template.readline().strip()
                    while not is_marker(line):
                        splut = line.split(":")
                        if len(splut) == 2:
                            colnum = int(splut[0])
                            colname = splut[1].strip()
                            self.colnames[colnum] = colname
                        line = template.readline()
                elif line.startswith("#csv"):
                    line = template.readline().strip()
                    self.csv = line
                    line = template.readline()
                elif line.startswith("#identifier"):
                    line = template.readline()
                    self.identifier = int(line.strip())
                    line = template.readline()
                elif line.startswith("#writepath"):
                    line = template.readline()
                    self.writepath = line.strip()
                    line = template.readline()
                elif line.startswith("#evaluation"):
                    self.evaluation = []
                    line = template.readline().strip()
                    while not is_marker(line):
                        l = line.strip()
                        if l:
                            self.evaluation.append(l)
                        line = template.readline()
                else:
                    line = template.readline()

        self.process_csv()
        self.process_evaluations()
        self.process_tex(comp=comp)


    def process_csv(self):
        to_extract = self.colnames
        index_num = self.identifier
        results = dict()
        with open(self.csv, newline='', encoding='latin-1') as csvfile:
            reading = csv.reader(csvfile)
            firstRow = next(reading)
            ncol = len(firstRow)
            for r in reading:
                index = r[index_num]
                print(f"Importing {index} from {self.csv}...")
                this_environment = syntaxtree.Environment()
                for i in range(ncol):
                    if i in to_extract:
                        variable_name = to_extract[i]
                        values = parse_json(r[i])
                        if values:
                            for k,v in values.items():
                                subname = self.delimiter.join([variable_name, k])
                                this_environment.process_literally(subname, v)
                        else:
                            this_environment.process_literally(variable_name, r[i])
                results[index] = this_environment
        self.students = results


    def process_evaluations(self):
        for num,environment in self.students.items():
            print(f"Processing evaluations for {num}...")
            for s in self.evaluation:
                try:
                    environment.process(s)
                except ZeroDivisionError as err:
                    raise LoadEvalError(f"ID {num}: {s}", err)
                except Exception as err:
                    raise LoadEvalError(f"ID {num}: {s}", err)

                    
    def process_tex(self,comp=False):
        class TexPlate(Template):
            delimiter = '@'
            idpattern = r'[a-z][_a-z0-9]*'
        template = TexPlate(self.tex)
        if not os.path.exists(self.writepath):
            os.makedirs(self.writepath)
        for num,environment in self.students.items():
            print(f"Writing LaTeX file for {num}...")
            subs = environment.get_subs(self.digits)
            text = template.safe_substitute(subs)
            filename = f"{num}_{self.suffix}.tex"
            write_to = f"{self.writepath}/{filename}"
            with open(write_to, "w") as wr:
                wr.write(text)
            if comp:
                print(f"Compiling {filename}...")
                subprocess.check_call(['latexmk', filename], cwd=self.writepath, stdout=subprocess.DEVNULL)
                print(f"Cleaning up after compiling {filename}...")
                subprocess.check_call(['latexmk', '-c', filename], cwd=self.writepath, stdout=subprocess.DEVNULL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dodgy LaTeX evaluation thingy")
    parser.add_argument("template", help="Specify the LaTeX template")
    parser.add_argument("--suffix", default="solutions", help="The suffix added to the LaTeX files" )
    parser.add_argument("--comp", action='store_true', help="If enabled, compile LaTeX files. Requires latexmk." )
    parser.add_argument("--dp", type=int, help="If specified, rounds all floats to DP decimal places")
    args = parser.parse_args()
    results = Importer(filename=args.template, suffix=args.suffix, comp=args.comp, digits=args.dp)
