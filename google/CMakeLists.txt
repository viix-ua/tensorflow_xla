
set (SOURCE_LIB_GOOGLE_PROTOBUF 

   google_arena.cc 
   #google_arenastring.cc 
   google_atomicops_internals_x86_gcc.cc 
   google_atomicops_internals_x86_msvc.cc 
   google_common.cc 
   google_generated_message_util.cc 
   google_once.cc 
   google_repeated_field.cc 
   
   #google_common.cc 
   )

add_library (lib_google_protobuf STATIC ${SOURCE_LIB_GOOGLE_PROTOBUF})

