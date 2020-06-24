git clone https://github.com/JohannesBuchner/MultiNest
cmake MultiNest
make -c MultiNest/build/
cd MultiNest/build && cmake .. && make && cd ../lib/
export LD_LIBRARY_PATH="`pwd`:$LD_LIBRARY_PATH"
cd ../..
