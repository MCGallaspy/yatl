test.exe: yatl_traits.hpp expression.hpp expr.cpp tensor.hpp
	g++ -std=c++14 -Wall -Werror -Wpedantic -O3 yatl_traits.hpp expression.hpp tensor.hpp expr.cpp -o test.exe

test: test.exe
	test.exe
