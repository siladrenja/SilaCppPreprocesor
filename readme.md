#Sila C++ Preprocessor
## intro
This is a python script I made to streamline some of C++ development, it is capable of applying attributes written in python to the code, and also splitting declarations from definitions into 2 separate files.

## splitting
It should respect folder structure, namespaces, and correctly split method and function declarations into ./build/header/{path}/name.hpp and definitions into ./build/src/{path}/name.cpp

## attributes
within the c++, attributes are added with a ``//@`` prefix, followed by the attribute name.

Attributes are python scripts within the folder (default ./attrib)

Each attribute is a function that takes in the string (the body of either a class or a method that it's attached to), and returns a string (what that class or method should be replaced with after editing)

## config
All configurations are written within config.py
The configurations are the following:

- file\_extension -> extension of files to look for (e.g. "cxx"), do not include the leading dot (.)
- attributes\_dir -> directory containing all of the attribute python scripts (e.g. "./attrib")
- target\_dir -> root directory of the folder in which to look for the file\_extension scripts (e.g. "./src")
- recursive -> whether or not to recursively look at subfolders of the target_dir for more files (e.g. true)
