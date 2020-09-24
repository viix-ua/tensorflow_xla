tensorflow_xla
===================
Tensorflow_xla is C++ standalone library for linear algebra & scientific computing. This is port of tensorflow-xla (linear-algebra level) from tensorflow-framework.

License and Copyright
---------------------
Copyright (C) 2020 - All rights reserved

All metadata is proprietary unless otherwise stated. 

License information for any other files is either explicitly stated or
defaults to proprietary.

# Build options

```
mkdir tensorflow_xla_build
cd tensorflow_xla_build
cmake -DCMAKE_BUILD_TYPE=Debug ../tensorflow_xla
cmake --build .
```

# Build

Build all
- 

```
git clone https://github.com/diixo/tensorflow_xla.git -b master
cd tensorflow_xla
cmake -H. -Bbuild
cmake --build build
```


