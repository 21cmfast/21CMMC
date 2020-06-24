git clone https://github.com/JohannesBuchner/MultiNest
cd MultiNest/build && cmake .. && make && cd ../lib/
export LD_LIBRARY_PATH="`pwd`:$LD_LIBRARY_PATH"
cd ../..
