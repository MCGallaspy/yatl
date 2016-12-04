/*
Hi Tor, I kept thinking about vectors, swizzling, and more general tensors.
I finally got the time to realize these ideas a little bit this past turkey slaughter festival.
Here it is, interested to hear your thoughts. I have annotated the source.

There are three (and a half) ideas to consider here:

1.  Generalized tensors, an idea which is not fully fleshed out yet.
    Vectors and matrices are special cases of tensors with ranks 1 and 2 respectively. [*]
    I access elements using tensor::get instead of ::operator[], since ::operator[]
    is constrained to take one parameter. I want to be able to access elements with
    syntax like my_vector.get(1) or my_matrix.get(3, 4) which isn't possible with
    operator[]. This also has the nice benefit that scalar types (tensors of rank 0)
    can be accessed with ::get(), for which operator[] provides no analog.

2.  Swizzling: don't copy a vector to swizzle it. In analogy to std::string_view,
    the data is already there, so just provide a wrapper to access it. For swizzling
    this means mapping indices to other indices.
    E.g. if you swizzle a 3D vector by mapping 2 -> 1 then if you access 0, 1, 2 of the 
    swizzled vector you are really accessing 0, 1, 1 of the underlying vector.
    I have managed to make a very optimizable implementation in my examples below,
    as measured by # of generated instructions. However a final analysis will require
    more testing.

3.  The core idea behind the swizzling implementation can be generalized to 
    expression templates, which are used in other linalg libs already. The basic idea
    is simple: use lazy evaluation to avoid copying temporary values in complex expressions.
    In the examples below a swizzled vector doesn't copy and transform its underlying vector,
    instead it provides a recipe for transforming the vector which is evaluated only when you
    access its elements. In essence, we trade off temporary copies for pointer dereferencing.
    The same idea can be applied to arbitrary operations -- below I implement operator+ for
    tensors as a proof of concept.

[*] Actually tensors and matrices are not exactly the same thing, and some libraries like tensorflow and Eigen's
    unsupported tensor module do not quite make the distinction.
    For instance a "non-square" matrix doesn't really correspond to a geometric tensor -- there isn't an elementary
    notion of a tensor whose indices have different dimensionalities.
    Basically there is a difference between, say, a rank-2 tensor and a vector of vectors.
*/

#include <tuple>
#include <type_traits>
#include <utility>

#include <stdio.h>

#include "tensor.hpp"

namespace yatl {

// Only (conceptually) defined for vectors. I'm not sure how swizzling would generalize to higher rank tensors.
template <typename expr_type>
class swizzled : public expression<swizzled<expr_type>> {
private:
  const expr_type m_vec;
  
  static constexpr int dims = get_dims<expr_type>::value;
  
  // The obvious first choice (to me) is a map, but this generates 100s of instructions.
  // Using an array, which neatly avoids heap-allocation, makes for much more
  // optimizable code.
  std::array<int, dims> m_from_to;
  
public:
  using value_type = typename get_value_type<swizzled<expr_type>>::type;

  template <typename pair_type>
  swizzled(const expr_type vec, pair_type&& pairs) : m_vec(vec) {
    // First make it the identity map
    for (int i=0; i<dims; ++i) {
      m_from_to[i] = i;
    }
    
    // Then set according to provided pairs
    for (const auto& pair : pairs) {
      m_from_to[pair.first] = pair.second;
    }
  }

  // We don't define a non-const version because it permits weird assignments.
  const auto get(int index) const {
    return m_vec.get(m_from_to[index]);
  }
};

template <typename inner>
struct get_dims<swizzled<inner>> {
  static constexpr int value = get_dims<inner>::value;
};

// Represents an arbitrary binary operation.
// Here for the first time we encounter an issue with dangling references.
// binary_ops can be nested, but one must be careful that an honest binary_op is copied and not the
// base class expression<binary_op<...>>, otherwise there will be issues.
template <typename left_t, typename right_t, typename op_t>
class binary_op : public expression<binary_op<left_t, right_t, op_t>> {
private:
  const left_t m_left;
  const right_t m_right;
  const op_t m_op;

public:
  using value_type = typename get_value_type<binary_op<left_t, right_t, op_t>>::type;
  
  binary_op(const left_t left, const right_t right, const op_t& op) :
    m_left(left), m_right(right), m_op(op) {}

  // Again we don't define a non-const version because it permits weird assignments.
  // What should be the meaning of assigning to an expression? In cases where an expression
  // should be evaluated into a real tensor, i.e. a distinct (transformed) copy of its source,
  // we would like to make this explicit.
  template <typename... index_types>
  const auto get(index_types&&... inds) const {
    return m_op(m_left.get(inds...), m_right.get(inds...));
  }
};

template <typename value_type, int dims = 3>
using vector = tensor<value_type, dims, 1>;

// Promoter eases the dangling reference problem... basically it can be used to say that an expression should be
// "promoted" to the referenced "inner" type if the inner type supports copy semantics.
// That limits the number of pathological expressions it is possible to create.
// However you're still in trouble if you create a temporary tensor.
template <typename inner>
struct promoter;

template <typename inner>
using promoter_t = typename promoter<inner>::type;

template <typename inner>
struct promoter<expression<inner>> {
    using type = std::conditional_t<std::is_copy_constructible<inner>::value, inner, expression<inner>>;
};


// This and operator+ below cut out some boilerplate, but in order to accept arbitrary expressions we open
// ourselves up to the dangling reference danger of "naked" expressions.
// In order to avoid it, we'll "promote" our parameters where possible.
template <typename expr_type, typename pairs_type>
auto make_swizzled(expression<expr_type> vec, pairs_type&& pairs) {
  using inner = promoter_t<expression<expr_type>>;
  using swizzled_type = swizzled<inner>;
  return swizzled_type(vec, std::forward<pairs_type>(pairs));
}

template <typename left_t, typename right_t>
auto operator+(expression<left_t> left, expression<right_t> right) {
  using left_vt = typename get_value_type<left_t>::type;
  using right_vt = typename get_value_type<right_t>::type;
  
  auto plus = [](const left_vt& lft_v, const right_vt& rgt_v) {
    return lft_v + rgt_v;
  };

  using left_res_t = promoter_t<expression<left_t>>;
  using right_res_t = promoter_t<expression<right_t>>;
  using binary_op_type = binary_op<left_res_t, right_res_t, decltype(plus)>;
  return binary_op_type(left, right, plus);
}

// Finally some helper template definitions
template <typename left, typename right, typename op>
class get_value_type<binary_op<left, right, op>> {
  using left_vt  = typename left::value_type;
  using right_vt = typename right::value_type;

public:
  using type = std::result_of_t<op(left_vt, right_vt)>;
};

// ...
template <typename expr_type>
struct get_value_type<swizzled<expr_type>> {
  using type = typename expr_type::value_type;
};

} // namespace yatl

template <typename value_type, int rows, int cols>
using matrix = yatl::vector<yatl::vector<value_type, cols>, rows>;

int main() {
  // The basics
  yatl::vector<int> my_3vec(42);
  printf("%d\n", my_3vec.get(0));
  
  my_3vec.get(1) = 7;
  printf("%d\n", my_3vec.get(1));
  
  // Can't assign to constant.
  // expr.get(0) = 5;
  
  // Maps index 2 into 1.
  std::array<std::pair<int, int>, 1> pairs{ std::make_pair(2, 1) };
  auto bar = yatl::make_swizzled(my_3vec, pairs);
  printf("%d\n", bar.get(2));
  
  // And you can swizzle swizzled stuff.
  auto baz = yatl::make_swizzled(bar, pairs);
  printf("%d\n", baz.get(0));
  
  // With a swizzled vector, it seems weird to permit assignment as in the example below.
  // bar.get(1) = 3;
  // printf("%d", bar.get(2)); 

  // Just for grins
  yatl::tensor<double, 3, 2> my_matrix(1.2);
  printf("%f\n", my_matrix.get(1, 1));
  
  // Scalar types just work, but there's sort of a weird syntax since the dimensionality doesn't matter.
  yatl::tensor<int, 3, 0> scalar(99);
  printf("%d\n", scalar.get());
  
  yatl::tensor<double, 0, 0> scalar2{2.2};
  printf("%f\n", scalar2.get());
  
  // Expressions can be chained arbitrarily, as is done implicitly below.
  yatl::vector<int, 3> another_3vec(9);
  auto result = my_3vec + another_3vec + bar;
  for (int i=0; i<3; ++i) {
    printf("%d\n", result.get(i));
  }
  
  /*
  // Temporary tensors are a problem, though... leaves a dangling reference.
  // Unsure how to deal with this issue.
  auto bad = my_3vec + yatl::vector<int, 3>(0);
  printf("%d\n", bad.get(1));
  */
  
  yatl::vector<int, 3> vec_init_tester{5, 6, 7};
  for (int i=0; i<3; ++i) {
    printf("%d\n", vec_init_tester.get(i));
  }
  
  yatl::tensor<int, 3, 2> matrix_init_tester{{5, 6, 7},
                                             {7, 8, 9},
                                             {3, 4, 1}};
  for (int i=0; i<3; ++i) {
    printf("%d, %d, %d\n",
           matrix_init_tester.get(0, i),
           matrix_init_tester.get(1, i),
           matrix_init_tester.get(2, i));
  }
  
  yatl::tensor<double, 2, 2> matrix_init_tester2{{5.2, 6},
                                                 {7  , 8}};
  for (int i=0; i<2; ++i) {
    printf("%f, %f\n",
           matrix_init_tester2.get(0, i),
           matrix_init_tester2.get(1, i));
  }
  
  yatl::tensor<char, 2, 3>
    rank3_char_tensor
    {
      {
        {'a', 'b'},
        {'c', 'd'}
      },
      {
        {'e', 'f'}, 
        {'g', 'h'}
      }
    };
  
  for (const auto& elm : rank3_char_tensor) {
    printf("%c, ", elm);
  }
  printf("\n");
  
  yatl::tensor<int, 2, 4>
    rank4_int_tensor
    {
      {
        {
          {15, 14},
          {13, 12}
        },
        {
          {11, 10}, 
          { 9,  8}
        }
      },
      {
        {
          { 7,  6},
          { 5,  4}
        },
        {
          { 3,  2}, 
          { 1,  0}
        }
      }
    };
  
  for (const auto& elm : rank4_int_tensor) {
    printf("%d, ", elm);
  }
  printf("\n");
  
  yatl::vector<int, 3> a = {1, 2, 3};
  matrix<int, 2, 3> rect;
  rect.get(0) = std::move(a);
  rect.get(1) = yatl::vector<int, 3>{4, 5, 6};
  for (const auto& row : rect) {
    for (const auto& elm : row) {
      printf("%d, ", elm);
    }
    printf("\n");
  }

  matrix<int, 3, 2> rect2 {
    {1, 2},
    {3, 4},
    {5, 6}
  };
  
  printf("%d, %d\n", rect2.get(0).get(0), rect2.get(0).get(1));
  printf("%d, %d\n", rect2.get(1).get(0), rect2.get(1).get(1));
  printf("%d, %d\n", rect2.get(2).get(0), rect2.get(2).get(1));
}