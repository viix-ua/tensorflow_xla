
#ifndef GOOGLE_PROTOBUF_REPEATED_PTR_ITERATOR_H__
#define GOOGLE_PROTOBUF_REPEATED_PTR_ITERATOR_H__

namespace google {

namespace protobuf {

namespace internal {

// STL-like iterator implementation for RepeatedPtrField.  You should not
// refer to this class directly; use RepeatedPtrField<T>::iterator instead.
//
// The iterator for RepeatedPtrField<T>, RepeatedPtrIterator<T>, is
// very similar to iterator_ptr<T**> in util/gtl/iterator_adaptors.h,
// but adds random-access operators and is modified to wrap a void** base
// iterator (since RepeatedPtrField stores its array as a void* array and
// casting void** to T** would violate C++ aliasing rules).
//
// This code based on net/proto/proto-array-internal.h by Jeffrey Yasskin
// (jyasskin@google.com).
template<typename Element>
class RepeatedPtrIterator
    : public std::iterator<
          std::random_access_iterator_tag, Element>
{
 public:
  typedef RepeatedPtrIterator<Element> iterator;
  typedef std::iterator<
          std::random_access_iterator_tag, Element> superclass;

  // Shadow the value_type in std::iterator<> because const_iterator::value_type
  // needs to be T, not const T.
  typedef typename remove_const<Element>::type value_type;

  // Let the compiler know that these are type names, so we don't have to
  // write "typename" in front of them everywhere.
  typedef typename superclass::reference reference;
  typedef typename superclass::pointer pointer;
  typedef typename superclass::difference_type difference_type;

  RepeatedPtrIterator() : it_(NULL) {}
  explicit RepeatedPtrIterator(void* const* it) : it_(it) {}

  // Allow "upcasting" from RepeatedPtrIterator<T**> to
  // RepeatedPtrIterator<const T*const*>.
  template<typename OtherElement>
  RepeatedPtrIterator(const RepeatedPtrIterator<OtherElement>& other)
      : it_(other.it_) {
    // Force a compiler error if the other type is not convertible to ours.
    if (false) {
      implicit_cast<Element*, OtherElement*>(0);
    }
  }

  // dereferenceable
  reference operator*() const { return *reinterpret_cast<Element*>(*it_); }
  pointer   operator->() const { return &(operator*()); }

  // {inc,dec}rementable
  iterator& operator++() { ++it_; return *this; }
  iterator  operator++(int) { return iterator(it_++); }
  iterator& operator--() { --it_; return *this; }
  iterator  operator--(int) { return iterator(it_--); }

  // equality_comparable
  bool operator==(const iterator& x) const { return it_ == x.it_; }
  bool operator!=(const iterator& x) const { return it_ != x.it_; }

  // less_than_comparable
  bool operator<(const iterator& x) const { return it_ < x.it_; }
  bool operator<=(const iterator& x) const { return it_ <= x.it_; }
  bool operator>(const iterator& x) const { return it_ > x.it_; }
  bool operator>=(const iterator& x) const { return it_ >= x.it_; }

  // addable, subtractable
  iterator& operator+=(difference_type d) {
    it_ += d;
    return *this;
  }
  friend iterator operator+(iterator it, const difference_type d) {
    it += d;
    return it;
  }
  friend iterator operator+(const difference_type d, iterator it) {
    it += d;
    return it;
  }
  iterator& operator-=(difference_type d) {
    it_ -= d;
    return *this;
  }
  friend iterator operator-(iterator it, difference_type d) {
    it -= d;
    return it;
  }

  // indexable
  reference operator[](difference_type d) const { return *(*this + d); }

  // random access iterator
  difference_type operator-(const iterator& x) const { return it_ - x.it_; }

 private:
  template<typename OtherElement>
  friend class RepeatedPtrIterator;

  // The internal iterator.
  void* const* it_;
};

// Provide an iterator that operates on pointers to the underlying objects
// rather than the objects themselves as RepeatedPtrIterator does.
// Consider using this when working with stl algorithms that change
// the array.
// The VoidPtr template parameter holds the type-agnostic pointer value
// referenced by the iterator.  It should either be "void *" for a mutable
// iterator, or "const void* const" for a constant iterator.
template <typename Element, typename VoidPtr>
class RepeatedPtrOverPtrsIterator
    : public std::iterator<std::random_access_iterator_tag, Element> {
 public:
  typedef RepeatedPtrOverPtrsIterator<Element, VoidPtr> iterator;
  typedef std::iterator<std::random_access_iterator_tag, Element> superclass;

  // Shadow the value_type in std::iterator<> because const_iterator::value_type
  // needs to be T, not const T.
  typedef typename remove_const<Element>::type value_type;

  // Let the compiler know that these are type names, so we don't have to
  // write "typename" in front of them everywhere.
  typedef typename superclass::reference reference;
  typedef typename superclass::pointer pointer;
  typedef typename superclass::difference_type difference_type;

  RepeatedPtrOverPtrsIterator() : it_(NULL) {}
  explicit RepeatedPtrOverPtrsIterator(VoidPtr* it) : it_(it) {}

  // dereferenceable
  reference operator*() const { return *reinterpret_cast<Element*>(it_); }
  pointer   operator->() const { return &(operator*()); }

  // {inc,dec}rementable
  iterator& operator++() { ++it_; return *this; }
  iterator  operator++(int) { return iterator(it_++); }
  iterator& operator--() { --it_; return *this; }
  iterator  operator--(int) { return iterator(it_--); }

  // equality_comparable
  bool operator==(const iterator& x) const { return it_ == x.it_; }
  bool operator!=(const iterator& x) const { return it_ != x.it_; }

  // less_than_comparable
  bool operator<(const iterator& x) const { return it_ < x.it_; }
  bool operator<=(const iterator& x) const { return it_ <= x.it_; }
  bool operator>(const iterator& x) const { return it_ > x.it_; }
  bool operator>=(const iterator& x) const { return it_ >= x.it_; }

  // addable, subtractable
  iterator& operator+=(difference_type d) {
    it_ += d;
    return *this;
  }
  friend iterator operator+(iterator it, difference_type d) {
    it += d;
    return it;
  }
  friend iterator operator+(difference_type d, iterator it) {
    it += d;
    return it;
  }
  iterator& operator-=(difference_type d) {
    it_ -= d;
    return *this;
  }
  friend iterator operator-(iterator it, difference_type d) {
    it -= d;
    return it;
  }

  // indexable
  reference operator[](difference_type d) const { return *(*this + d); }

  // random access iterator
  difference_type operator-(const iterator& x) const { return it_ - x.it_; }

 private:
  template<typename OtherElement>
  friend class RepeatedPtrIterator;

  // The internal iterator.
  VoidPtr* it_;
};

}  // namespace internal

}

}

#endif 
