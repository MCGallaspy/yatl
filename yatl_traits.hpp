#ifndef YATL_TRAITS_HPP
#define YATL_TRAITS_HPP

namespace yatl {

// A little helper that we'll make specializations for.
// Cuts down on some visual clutter in the definitions.
template <typename>
struct get_value_type;

// Metafunction whose specializations return the associated dimension of a tensor.
template <typename>
struct get_dims;

} // namespace yatl
#endif // YATL_TRAITS_HPP