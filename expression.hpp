#ifndef EXPRESSION_HPP
#define EXPRESSION_HPP
#include "yatl_traits.hpp"

// An expression wraps non-copyable objects (for us, tensors) to give us cheap copy semantics.
// However we must be careful to avoid using "naked expressions" -- it will result in a dangling pointer
// if the expression outlives the object it wraps, which is often and quite likely with temporaries.
// 
// So when *should* we use expressions? A function can be defined for arbitrary expressions as follows, but
// care must be taken not to *return* naked expressions:
// 
//   template <typename inner>
//   int function(expression<inner> param);
//

namespace yatl {

template <typename base_type>
class expression {
private:
  const base_type* m_base;
  
public:
  expression() : m_base(static_cast<const base_type*>(this)) {}
  expression(const expression&) = default;
  expression(expression&&) = default;
  expression& operator=(const expression&) = default;
  expression& operator=(expression&&) = default;

  using value_type = typename get_value_type<base_type>::type;

  template <typename... ind_ts>
  const value_type get(ind_ts&&... inds) const {
    return m_base->get(inds...);
  }

  // Permits static_casts back to the base type.
  operator const base_type& () const {
    return *m_base;
  }
};

//***********************
// Helper specializations
//***********************

template <typename inner>
struct get_dims<expression<inner>> {
  static constexpr int value = get_dims<inner>::value;
};

template <typename inner>
struct get_value_type<expression<inner>> {
  using type = typename get_value_type<inner>::type;
};

} // namespace yatl
#endif // EXPRESSION_HPP
