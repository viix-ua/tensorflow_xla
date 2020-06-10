
set (SOURCE_LIB_GOOGLE_PROTOBUF 

   google_arena.cc 
   #google_arenastring.cc 
   google_atomicops_internals_x86_msvc.cc 
   google_common.cc 
   google_generated_message_util.cc 
   google_once.cc 
   google_repeated_field.cc 
   
   #google_common.cc 
   )


add_library (lib_google_protobuf STATIC ${SOURCE_LIB_GOOGLE_PROTOBUF})


#HBAdditionalFiles google_protobuf :
#
#   google_arena.h 
#   google_arenastring.h 
#   google_atomicops.h 
#   google_atomicops_internals_x86_msvc.h 
#   google_atomic_sequence_num.h 
#   google_callback.h 
#   google_casts.h 
#   google_common.h 
#   google_fastmem.h 
#   google_generated_message_util.h 
#   google_has_bits.h 
#   google_logging.h 
#   google_macros.h 
#   google_mutex.h 
#   google_once.h 
#   google_platform_macros.h 
#   google_port.h 
#   google_repeated_field.h 
#   google_scoped_ptr.h 
#   google_template_util.h 
#   google_type_traits.h 

#   #google_common.h 
#   #google_logging.h 
#   ;
