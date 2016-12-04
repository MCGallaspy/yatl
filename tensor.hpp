#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <array>
#include <initializer_list>

#include "expression.hpp"

namespace yatl {

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

template <int rank, typename value_type>
struct nested_init_list {
  using type = std::initializer_list<
    typename nested_init_list<rank - 1, value_type>::type
  >;
};

template <typename value_type>
struct nested_init_list<1, value_type> {
  using type = std::initializer_list<value_type>;
};

// A special case, not reachable by recursion but necessary nonetheless
template <typename value_type>
struct nested_init_list<0, value_type> {
  using type = std::initializer_list<value_type>;
};

template <int rank, typename value_type>
using nested_init_list_t = typename nested_init_list<rank, value_type>::type;

template <typename nested_list, int dims, typename = void>
struct coord_arr_maker;

template <typename value_type, int dims, int rank>
struct coord_arr_maker<nested_init_list<rank, value_type>, dims> {
  static
  auto
  make(nested_init_list_t<rank, value_type> list) {
    std::array<value_type, pow(dims, rank)> arr;
    int i = 0;
    for (auto& sublist : list) {
      auto subseq = coord_arr_maker<nested_init_list<rank - 1, value_type>, dims>::make(sublist);
      int j = 0;
      for (auto& elm : subseq) {
        constexpr int base = pow(dims, rank - 1);
        const int index = base * i + j;
        // Todo: separate this into copy-assign and move-assign cases as below.
        // arr[index] = elm;
        arr[index] = std::move(const_cast<value_type&>(elm));
        ++j;
      }
      ++i;
    }
    return arr;
  }
};

template <typename value_type_, int dims>
struct coord_arr_maker<nested_init_list<1, value_type_>, dims> {
private:
  template <typename value_type>
  static auto
  make(std::initializer_list<value_type> list, std::false_type) {
    std::array<value_type, dims> arr;
    int i = 0;
    for (const auto& elm : list) {
      arr[i] = std::move(const_cast<value_type&>(elm));
      ++i;
    }
    return arr;
  }

  template <typename value_type>
  static auto
  make(std::initializer_list<value_type> list, std::true_type) {
    std::array<value_type, dims> arr;
    int i = 0;
    for (const auto& elm : list) {
      arr[i] = elm;
      ++i;
    }
    return arr;
  }

public:
  static auto
  make(std::initializer_list<value_type_> list) {
    return make(list, std::is_copy_assignable<value_type_>{});
  }
};

template <typename value_type, int dims>
struct coord_arr_maker<nested_init_list<0, value_type>, dims> {
  static
  auto
  make(nested_init_list_t<0, value_type> list) {
    return std::array<value_type, 1>{*list.begin()};
  }
};

} // anonymous namespace

// value_type_ : int, double, etc. Should support arithmetic.
// dims        : the dimensionality, e.g. 3 for normal Euclidean space
// rank        : 0 is a scalar, 1 is a vector, 2 a matrix, etc.
template <typename value_type_, int dims = 3, int rank = 1>
class tensor : public expression<tensor<value_type_, dims, rank>> {
  // Use an array, becuase heap allocation is the enemy of optimization.
  std::array<value_type_, pow(dims, rank)> m_coords;

public:
  tensor(const tensor& copy)           = delete;
  tensor& operator=(const tensor& rhs) = delete;
  
  tensor()                        = default;
  tensor(tensor&& rhs)            = default;
  tensor& operator=(tensor&& rhs) = default;
  
  using value_type = value_type_;
  static constexpr int dimensionality = dims;

  // Initializes every element by copying proto
  tensor(const value_type& proto) {
    for (auto& elem : m_coords) {
      elem = proto;
    }
  }

  tensor(nested_init_list_t<rank, value_type> list) :
    m_coords(
      coord_arr_maker<nested_init_list<rank, value_type>, dims>::make(list)
    ) {}

  // The non-const accessor is hackily defined in terms of the const operator by casting
  // away the const-ness. See const version of explanation.
  template <typename... ind_ts>
  value_type& operator()(ind_ts&&... inds) {
    using tensor_type = tensor<value_type, dims, rank>;
    return const_cast<value_type&>(static_cast<const tensor_type&>(*this)(inds...));
  }
  
  // I don't really care about the types of indices.
  // The only thing that matters is the # of parameters, which are used to index into a flat array.
  template <typename... ind_ts>
  const value_type& operator()(ind_ts&&... inds) const {
    static_assert(sizeof...(ind_ts) == rank, "The number of indices must match the rank of the tensor.");
    auto index = linearize<dims>(inds...);
    return m_coords[index];
  }
  
  auto begin() const {
    return m_coords.begin();
  }
  
  auto end() const {
    return m_coords.end();
  }
};

//***********************
// Helper specializations
//***********************

template <typename value_type, int dims, int rank>
struct get_dims<tensor<value_type, dims, rank>> {
  static constexpr int value = dims;
};

template <typename value_type, int dims, int rank>
struct get_value_type<tensor<value_type, dims, rank>> {
  using type = value_type;
};

} // namespace yatl
#endif // TENSOR_HPP
