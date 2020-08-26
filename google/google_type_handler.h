
#ifndef GOOGLE_PROTOBUF_TYPE_HANDLER_H__
#define GOOGLE_PROTOBUF_TYPE_HANDLER_H__

#include <iterator>
#include <limits>
#include <string>

// Forward-declare these so that we can make them friends.
namespace google {

namespace protobuf {

namespace internal
{

template <typename GenericType>
class GenericTypeHandler {
 public:
  typedef GenericType Type;
#if LANG_CXX11
  static const bool Moveable = false;
#endif

  static inline GenericType* New(Arena* arena) {
    return ::google::protobuf::Arena::CreateMaybeMessage<Type>(
        arena, static_cast<GenericType*>(0));
  }
  static inline GenericType* NewFromPrototype(
      const GenericType* prototype, ::google::protobuf::Arena* arena = NULL);
  static inline void Delete(GenericType* value, Arena* arena) {
    if (arena == NULL) {
      delete value;
    }
  }
  static inline ::google::protobuf::Arena* GetArena(GenericType* value) {
    return ::google::protobuf::Arena::GetArena<Type>(value);
  }
  static inline void* GetMaybeArenaPointer(GenericType* value) {
    return ::google::protobuf::Arena::GetArena<Type>(value);
  }

  static inline void Clear(GenericType* value) { value->Clear(); }
  GOOGLE_ATTRIBUTE_NOINLINE static void Merge(const GenericType& from,
                                       GenericType* to);
  static inline size_t SpaceUsedLong(const GenericType& value) {
    return value.SpaceUsedLong();
  }
  //static inline const Type& default_instance() {
  //  return Type::default_instance();
  //}
};

template <typename GenericType>
GenericType* GenericTypeHandler<GenericType>::NewFromPrototype(
    const GenericType* /* prototype */, ::google::protobuf::Arena* arena) {
  return New(arena);
}
template <typename GenericType>
void GenericTypeHandler<GenericType>::Merge(const GenericType& from,
                                            GenericType* to) {
  to->MergeFrom(from);
}

// NewFromPrototype() and Merge() are not defined inline here, as we will need
// to do a virtual function dispatch anyways to go from Message* to call
// New/Merge.
/*
template<>
MessageLite* GenericTypeHandler<MessageLite>::NewFromPrototype(
    const MessageLite* prototype, google::protobuf::Arena* arena);
template<>
inline google::protobuf::Arena* GenericTypeHandler<MessageLite>::GetArena(
    MessageLite* value) {
  return value->GetArena();
}
template<>
inline void* GenericTypeHandler<MessageLite>::GetMaybeArenaPointer(
    MessageLite* value) {
  return value->GetMaybeArenaPointer();
}
template <>
void GenericTypeHandler<MessageLite>::Merge(const MessageLite& from,
                                            MessageLite* to);
*/
template<>
inline void GenericTypeHandler<string>::Clear(string* value) {
  value->clear();
}
template<>
void GenericTypeHandler<string>::Merge(const string& from,
                                       string* to);

// Declarations of the specialization as we cannot define them here, as the
// header that defines ProtocolMessage depends on types defined in this header.
#define DECLARE_SPECIALIZATIONS_FOR_BASE_PROTO_TYPES(TypeName)                 \
    template<>                                                                 \
    TypeName* GenericTypeHandler<TypeName>::NewFromPrototype(                  \
        const TypeName* prototype, google::protobuf::Arena* arena);                      \
    template<>                                                                 \
    google::protobuf::Arena* GenericTypeHandler<TypeName>::GetArena(                     \
        TypeName* value);                                                      \
    template<>                                                                 \
    void* GenericTypeHandler<TypeName>::GetMaybeArenaPointer(                  \
        TypeName* value);

// Message specialization bodies defined in message.cc. This split is necessary
// to allow proto2-lite (which includes this header) to be independent of
// Message.
DECLARE_SPECIALIZATIONS_FOR_BASE_PROTO_TYPES(Message)


#undef DECLARE_SPECIALIZATIONS_FOR_BASE_PROTO_TYPES
/*
template <>
inline const MessageLite& GenericTypeHandler<MessageLite>::default_instance() {
  // Yes, the behavior of the code is undefined, but this function is only
  // called when we're already deep into the world of undefined, because the
  // caller called Get(index) out of bounds.
  MessageLite* null = NULL;
  return *null;
}
*/

//template <>
//inline const Message& GenericTypeHandler<Message>::default_instance() {
//  // Yes, the behavior of the code is undefined, but this function is only
//  // called when we're already deep into the world of undefined, because the
//  // caller called Get(index) out of bounds.
//  Message* null = NULL;
//  return *null;
//}


class StringTypeHandler {
 public:
  typedef string Type;
#if LANG_CXX11
  static const bool Moveable =
      std::is_move_constructible<Type>::value &&
      std::is_move_assignable<Type>::value;
#endif

  static inline string* New(Arena* arena) {
    return Arena::Create<string>(arena);
  }
#if LANG_CXX11
  static inline string* New(Arena* arena, string&& value) {
    return Arena::Create<string>(arena, std::move(value));
  }
#endif
  static inline string* NewFromPrototype(const string*,
                                         ::google::protobuf::Arena* arena) {
    return New(arena);
  }
  static inline ::google::protobuf::Arena* GetArena(string*) {
    return NULL;
  }
  static inline void* GetMaybeArenaPointer(string* /* value */) {
    return NULL;
  }
  static inline void Delete(string* value, Arena* arena) {
    if (arena == NULL) {
      delete value;
    }
  }
  static inline void Clear(string* value) { value->clear(); }
  static inline void Merge(const string& from, string* to) { *to = from; }
  //static inline const Type& default_instance() {
  //  return ::google::protobuf::internal::GetEmptyString();
  //}
  //static size_t SpaceUsedLong(const string& value)  {
  //  return sizeof(value) + StringSpaceUsedExcludingSelfLong(value);
  //}
};

}  // namespace internal

}

}

#endif
