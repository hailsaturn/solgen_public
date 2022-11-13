# Parsing script for making LaTeX files

Fair warning: this is undocumented and a little untested. I've added it here to share in a Reddit thread.

- The file `syntaxtree.py` defines the parser using [PyParsing](https://pyparsing-docs.readthedocs.io/) and has the components to define a grammar, syntax tree, scope and environment. 

- The file `solgen.py` is used to run the script without thinking. Usage:
```
python3 solgen.py example.tex
```
If you have `latexmk` installed, you can compile them immediately:
```
python3 solgen.py example.tex --comp
```
For more, write `python3 solgen.py -h`. The features are minimal at the moment.

- The file `example.tex` has example usage of the template file. It depends on `inputs.csv`.
	- The #colnames header specifies names for columns in the CSV file. Any column that is omitted is not imported. 
	- The #identifier header specifies the column number with the unique identifier that row (student number in my most usual use case)
	- The #writepath header specifies the directory that the files will be written to. Will be created if it doesn't exist.
	- The #csv header specifies the name of the CSV to import variables from
	- The #evaluation header specifies the start of a sequence of variables/functions to be defined
	- Once "\documentclass" is detected in a line, it is assumed that the file from that point on is the desired LaTeX file. 

This is still a complete work in progress, but I'm sharing the minimal version by request.
