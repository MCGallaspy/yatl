/*
Hi Tor, I kept thinking about vectors, swizzling, and more general tensors.
I finally got the time to realize these ideas a little but this past turkey slaughter festival.
Here it is, interested to hear your thoughts. I have annotated the source.

There are three (and a half) ideas to consider here:

1a. Generalized tensors, an idea which is not fully fleshed out yet.
    Vectors and matrices are special cases of tensors with ranks 1 and 2 respectively. [*]
    I access elements using tensor::get instead of ::operator[], since ::operator[]
    is constrained to take one parameter. I want to be able to access elements with
    syntax like my_vector.get(1) or my_matrix.get(3, 4) which isn't possible with
    operator[]. This also has the nice benefit that scalar types (tensors of rank 0)
    can be accessed with ::get(), for which operator[] provides no analog.
    
1b. For element access one might also want my_matrix.get(3) to return a tensor
    of one rank lower, i.e. a vector... this idea is not explored yet.

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
    access its elements. In essence, we tradeoff temporary copies for pointer dereferencing.
    The same idea can be applied to arbitrary operations -- below I implement operator+ for
    tensors as a proof of concept.

[*] The "dimensionality" of a tensor is distinct from it's rank... I don't think this is the
"real math" term for it and I'm too lazy to look it up, but the dimensionality as I use the
terms represents the # of dimensions of the underlying vector space, e.g. tensors over
a Euclidean plane would have dimensionality 2 and over a space they would have dimensionality 3.
*/

#include <array>
#include <type_traits>
#include <unordered_map>
#include <utility>

#include <stdio.h>

namespace cool {

// This anonymous namespace just has some utilities I use.
namespace {
constexpr int pow(int base, int power) {
  return power == 0 ? 1 : base * pow(base, power - 1);
}

// Base case for below
template <int dims>
constexpr int linearize() {
	return 0;
}

// All tensor elements are stored in a flat array. Therefore we need a general function
// that maps expressions like ::get(3, 4), ::get(7), and ::get(1, 2, 1) into indices
// of a flat array. This is such a function.
// Given a parameter pack, return a number that indexes into a flat array of elements.
template <int dims, typename head, typename... tail>
constexpr int linearize(head&& i, tail&&... t) {
  return i + (dims * linearize<dims>(t...));
}

} // anonymous inner namespace


// A little helper that we'll make specializations for below for each expression type.
// Cuts down on some visual clutter in the definitions.
template <typename>
struct get_value_type;
  
template <typename>
struct get_dims;

// An expression wraps non-copyable objects (for us, tensors) to give us (comparatively) cheap copy semantics.
template <typename base_type>
class expression {
private:
  const base_type& m_base;
  
public:
  expression(const base_type& base) : m_base(base) {}
  expression(const expression& copy) : m_base(copy.m_base) {}
  expression& operator=(const expression& copy) = delete;
  expression() = delete;
  
  using value_type = typename get_value_type<base_type>::type;
  
  template <typename... ind_ts>
  const value_type get(ind_ts&&... inds) const {
    return m_base.get(inds...);
  }
  
};

// value_type_ : int, double, etc. Should support arithmetic.
// dims        : the dimensionality, e.g. 3 for Euclidean space
// rank        : 0 is a scalar, 1 is a vector, 2 a matrix, etc.
template <typename value_type_, int dims = 3, int rank = 1>
class tensor : expression<tensor<value_type_, dims, rank>> {
  // Use an array instead of a vector, becuase heap allocation is the enemy of optimization.
  std::array<value_type_, pow(dims, rank)> m_coords;

public:
  // Just to prove that we're not copying anywhere!
  tensor(const tensor& copy) = delete;
  tensor(tensor&& copy) = delete;
  
  using value_type = value_type_;
  static constexpr int dimensionality = dims;
  
  tensor() = delete;
  // Initializes every element by copying proto
  tensor(const value_type& proto) : expression<tensor<value_type_, dims, rank>>(*this) {
    printf("making a tensor\n");
    for (auto& elem : m_coords) {
      elem = proto;
    }
  }
  
  // The non-const accessor is hackily defined in terms of the const operator by casting
  // away the const-ness. See const version of explanation.
  template <typename... ind_ts>
  value_type& get(ind_ts&&... inds) {
    using tensor_type = tensor<value_type, dims, rank>;
    return const_cast<value_type&>(static_cast<const tensor_type&>(*this).get(inds...));
  }
  
  // I don't really care about the types of indices.
  // The only thing that matters is the # of parameters, which are used to index into a flat array.
  template <typename... ind_ts>
  const value_type& get(ind_ts&&... inds) const {
    static_assert(sizeof...(ind_ts) == rank, "The number of indices must match the rank of the tensor.");
    auto index = linearize<dims>(inds...);
    return m_coords[index];
  }
};

template <typename inner>
struct get_dims<expression<inner>> {
  static constexpr int value = get_dims<inner>::value;
};

template <typename value_type, int dims, int rank>
struct get_dims<tensor<value_type, dims, rank>> {
  static constexpr int value = dims;
};

// ******************************************************************************* 
// The next classes, swizzled and binary_op, are examples of expression templates.
// Initially I had something wikipedia-esque involving the curiously recurring template pattern (CRTP), but I found
// that it doesn't add anything... I can achieve minimal code repitition and still get lazy evaluation without it.
// I wonder if I am missing the point somehow? Needs further investigation.
// *******************************************************************************

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
  
  using base_expr_type = expression<swizzled<expr_type>>;
  
public:
  using value_type = typename get_value_type<swizzled<expr_type>>::type;

  template <typename pair_type>
  swizzled(const expr_type vec, pair_type&& pairs) : base_expr_type(*this), m_vec(vec) {
    printf("Making a swizzle\n");
    // First make it the identity map
    for (int i=0; i<dims; ++i) {
      m_from_to[i] = i;
    }
    
    // Then set according to provided pairs
    for (const auto& pair : pairs) {
      m_from_to[pair.first] = pair.second;
    }
  }
  
  swizzled(const swizzled&) = delete;
  swizzled(swizzled&&) = default;

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
// Probably more thought should be given to taking op_t by rvalue reference
template <typename left_t, typename right_t, typename op_t>
class binary_op : public expression<binary_op<left_t, right_t, op_t>> {
private:
  const left_t m_left;
  const right_t m_right;
  const op_t m_op;
  
  using base_expr_type = expression<binary_op<left_t, right_t, op_t>>;

public:
  using value_type = typename get_value_type<binary_op<left_t, right_t, op_t>>::type;
  
  binary_op(const left_t left, const right_t right, const op_t& op) :
    base_expr_type(*this), m_left(left), m_right(right), m_op(op) {
        printf("Making a binary op\n");
    }

  // Again we don't define a non-const version because it permits weird assignments.
  // What should be the meaning of assigning to an expression? In cases where an expression
  // should be evaluated into a real tensor, i.e. a distinct (transformed) copy of its source,
  // we would like to make this explicit.
  template <typename... index_types>
  const auto get(index_types&&... inds) const {
    printf("get`ing from a binary_op\n");
    return m_op(m_left.get(inds...), m_right.get(inds...));
  }
};

template <typename value_type, int dims = 3>
using vector = tensor<value_type, 3, 1>;

template <typename expr_type, typename pairs_type>
swizzled<expression<expr_type>>
make_swizzled(expression<expr_type> vec, pairs_type&& pairs) {
  using swizzled_type = swizzled<expression<expr_type>>;
  return swizzled_type(vec, std::forward<pairs_type>(pairs));
}

template <typename left_t, typename right_t>
auto operator+(expression<left_t> left, expression<right_t> right) {
  using left_vt = typename get_value_type<left_t>::type;
  using right_vt = typename get_value_type<right_t>::type;
  
  constexpr auto plus = [](const left_vt& lft_v, const right_vt& rgt_v) {
    return lft_v + rgt_v;
  };

  using binary_op_type = binary_op<expression<left_t>, expression<right_t>, decltype(plus)>;
  return binary_op_type(left, right, plus);
}

template <typename left, typename right, typename op>
class get_value_type<binary_op<left, right, op>> {
  using left_vt  = typename left::value_type;
  using right_vt = typename right::value_type;

public:
  using type = std::result_of_t<op(left_vt, right_vt)>;
};

template <typename inner>
struct get_value_type<expression<inner>> {
  using type = typename get_value_type<inner>::type;
};

template <typename expr_type>
struct get_value_type<swizzled<expr_type>> {
  using type = typename expr_type::value_type;
};

template <typename value_type_, int dims, int rank>
struct get_value_type<tensor<value_type_, dims, rank>> {
  using type = value_type_;  
};
  
} // namespace cool


int main() {
  // The basics
  
  cool::vector<int> my_3vec(42);
  printf("%d\n", my_3vec.get(0));
  
  my_3vec.get(1) = 7;
  printf("%d\n", my_3vec.get(1));
  
  // Can't assign to constant.
  // expr.get(0) = 5;
  
  // Maps index 2 into 1.
  std::array<std::pair<int, int>, 1> pairs{ std::make_pair(2, 1) };
  auto bar = cool::make_swizzled(my_3vec, pairs);
  printf("%d\n", bar.get(2));
  
  auto baz = cool::make_swizzled(bar, pairs);
  printf("%d\n", baz.get(0));
  // With a swizzled vector, it seems weird to permit assignment as in the two examples below.
  // bar.get(1) = 3;
  // printf("%d", bar.get(2)); 

  
  // Just for grins
  cool::tensor<double, 3, 2> my_matrix(1.2);
  printf("%f\n", my_matrix.get(1, 1));
  
  // Scalar types just work.
  cool::tensor<int, 3, 0> scalar(99);
  printf("%d\n", scalar.get());
  
  // Expressions can be chained... but in the final analysis my knowledge of assembly is too
  // weak to determine whether this is optimal.
  cool::vector<int, 3> another_3vec(9);
  auto result = my_3vec + another_3vec + bar;
  auto result0 = my_3vec + another_3vec;
  auto result1 = result0 + bar;
  for (int i=0; i<3; ++i) {
    printf("%d\n", result1.get(i));
  }
  printf("%d", result.get(0));
}