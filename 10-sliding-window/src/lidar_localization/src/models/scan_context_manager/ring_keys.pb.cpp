// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: ring_keys.proto

#include "lidar_localization/models/scan_context_manager/ring_keys.pb.h"

#include <algorithm>

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/extension_set.h>
#include <google/protobuf/wire_format_lite.h>
#include <google/protobuf/descriptor.h>
#include <google/protobuf/generated_message_reflection.h>
#include <google/protobuf/reflection_ops.h>
#include <google/protobuf/wire_format.h>
// @@protoc_insertion_point(includes)
#include <google/protobuf/port_def.inc>

PROTOBUF_PRAGMA_INIT_SEG
extern PROTOBUF_INTERNAL_EXPORT_ring_5fkeys_2eproto ::PROTOBUF_NAMESPACE_ID::internal::SCCInfo<0> scc_info_RingKey_ring_5fkeys_2eproto;
namespace scan_context_io {
class RingKeyDefaultTypeInternal {
 public:
  ::PROTOBUF_NAMESPACE_ID::internal::ExplicitlyConstructed<RingKey> _instance;
} _RingKey_default_instance_;
class RingKeysDefaultTypeInternal {
 public:
  ::PROTOBUF_NAMESPACE_ID::internal::ExplicitlyConstructed<RingKeys> _instance;
} _RingKeys_default_instance_;
}  // namespace scan_context_io
static void InitDefaultsscc_info_RingKey_ring_5fkeys_2eproto() {
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  {
    void* ptr = &::scan_context_io::_RingKey_default_instance_;
    new (ptr) ::scan_context_io::RingKey();
    ::PROTOBUF_NAMESPACE_ID::internal::OnShutdownDestroyMessage(ptr);
  }
}

::PROTOBUF_NAMESPACE_ID::internal::SCCInfo<0> scc_info_RingKey_ring_5fkeys_2eproto =
    {{ATOMIC_VAR_INIT(::PROTOBUF_NAMESPACE_ID::internal::SCCInfoBase::kUninitialized), 0, 0, InitDefaultsscc_info_RingKey_ring_5fkeys_2eproto}, {}};

static void InitDefaultsscc_info_RingKeys_ring_5fkeys_2eproto() {
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  {
    void* ptr = &::scan_context_io::_RingKeys_default_instance_;
    new (ptr) ::scan_context_io::RingKeys();
    ::PROTOBUF_NAMESPACE_ID::internal::OnShutdownDestroyMessage(ptr);
  }
}

::PROTOBUF_NAMESPACE_ID::internal::SCCInfo<1> scc_info_RingKeys_ring_5fkeys_2eproto =
    {{ATOMIC_VAR_INIT(::PROTOBUF_NAMESPACE_ID::internal::SCCInfoBase::kUninitialized), 1, 0, InitDefaultsscc_info_RingKeys_ring_5fkeys_2eproto}, {
      &scc_info_RingKey_ring_5fkeys_2eproto.base,}};

static ::PROTOBUF_NAMESPACE_ID::Metadata file_level_metadata_ring_5fkeys_2eproto[2];
static constexpr ::PROTOBUF_NAMESPACE_ID::EnumDescriptor const** file_level_enum_descriptors_ring_5fkeys_2eproto = nullptr;
static constexpr ::PROTOBUF_NAMESPACE_ID::ServiceDescriptor const** file_level_service_descriptors_ring_5fkeys_2eproto = nullptr;

const ::PROTOBUF_NAMESPACE_ID::uint32 TableStruct_ring_5fkeys_2eproto::offsets[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) = {
  ~0u,  // no _has_bits_
  PROTOBUF_FIELD_OFFSET(::scan_context_io::RingKey, _internal_metadata_),
  ~0u,  // no _extensions_
  ~0u,  // no _oneof_case_
  ~0u,  // no _weak_field_map_
  PROTOBUF_FIELD_OFFSET(::scan_context_io::RingKey, data_),
  ~0u,  // no _has_bits_
  PROTOBUF_FIELD_OFFSET(::scan_context_io::RingKeys, _internal_metadata_),
  ~0u,  // no _extensions_
  ~0u,  // no _oneof_case_
  ~0u,  // no _weak_field_map_
  PROTOBUF_FIELD_OFFSET(::scan_context_io::RingKeys, data_),
};
static const ::PROTOBUF_NAMESPACE_ID::internal::MigrationSchema schemas[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) = {
  { 0, -1, sizeof(::scan_context_io::RingKey)},
  { 6, -1, sizeof(::scan_context_io::RingKeys)},
};

static ::PROTOBUF_NAMESPACE_ID::Message const * const file_default_instances[] = {
  reinterpret_cast<const ::PROTOBUF_NAMESPACE_ID::Message*>(&::scan_context_io::_RingKey_default_instance_),
  reinterpret_cast<const ::PROTOBUF_NAMESPACE_ID::Message*>(&::scan_context_io::_RingKeys_default_instance_),
};

const char descriptor_table_protodef_ring_5fkeys_2eproto[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) =
  "\n\017ring_keys.proto\022\017scan_context_io\"\027\n\007Ri"
  "ngKey\022\014\n\004data\030\001 \003(\002\"2\n\010RingKeys\022&\n\004data\030"
  "\001 \003(\0132\030.scan_context_io.RingKey"
  ;
static const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable*const descriptor_table_ring_5fkeys_2eproto_deps[1] = {
};
static ::PROTOBUF_NAMESPACE_ID::internal::SCCInfoBase*const descriptor_table_ring_5fkeys_2eproto_sccs[2] = {
  &scc_info_RingKey_ring_5fkeys_2eproto.base,
  &scc_info_RingKeys_ring_5fkeys_2eproto.base,
};
static ::PROTOBUF_NAMESPACE_ID::internal::once_flag descriptor_table_ring_5fkeys_2eproto_once;
const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable descriptor_table_ring_5fkeys_2eproto = {
  false, false, descriptor_table_protodef_ring_5fkeys_2eproto, "ring_keys.proto", 111,
  &descriptor_table_ring_5fkeys_2eproto_once, descriptor_table_ring_5fkeys_2eproto_sccs, descriptor_table_ring_5fkeys_2eproto_deps, 2, 0,
  schemas, file_default_instances, TableStruct_ring_5fkeys_2eproto::offsets,
  file_level_metadata_ring_5fkeys_2eproto, 2, file_level_enum_descriptors_ring_5fkeys_2eproto, file_level_service_descriptors_ring_5fkeys_2eproto,
};

// Force running AddDescriptors() at dynamic initialization time.
PROTOBUF_ATTRIBUTE_INIT_PRIORITY static ::PROTOBUF_NAMESPACE_ID::internal::AddDescriptorsRunner dynamic_init_dummy_ring_5fkeys_2eproto(&descriptor_table_ring_5fkeys_2eproto);
namespace scan_context_io {

// ===================================================================

class RingKey::_Internal {
 public:
};

RingKey::RingKey(::PROTOBUF_NAMESPACE_ID::Arena* arena)
  : ::PROTOBUF_NAMESPACE_ID::Message(arena),
  data_(arena) {
  SharedCtor();
  RegisterArenaDtor(arena);
  // @@protoc_insertion_point(arena_constructor:scan_context_io.RingKey)
}
RingKey::RingKey(const RingKey& from)
  : ::PROTOBUF_NAMESPACE_ID::Message(),
      data_(from.data_) {
  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
  // @@protoc_insertion_point(copy_constructor:scan_context_io.RingKey)
}

void RingKey::SharedCtor() {
}

RingKey::~RingKey() {
  // @@protoc_insertion_point(destructor:scan_context_io.RingKey)
  SharedDtor();
  _internal_metadata_.Delete<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

void RingKey::SharedDtor() {
  GOOGLE_DCHECK(GetArena() == nullptr);
}

void RingKey::ArenaDtor(void* object) {
  RingKey* _this = reinterpret_cast< RingKey* >(object);
  (void)_this;
}
void RingKey::RegisterArenaDtor(::PROTOBUF_NAMESPACE_ID::Arena*) {
}
void RingKey::SetCachedSize(int size) const {
  _cached_size_.Set(size);
}
const RingKey& RingKey::default_instance() {
  ::PROTOBUF_NAMESPACE_ID::internal::InitSCC(&::scc_info_RingKey_ring_5fkeys_2eproto.base);
  return *internal_default_instance();
}


void RingKey::Clear() {
// @@protoc_insertion_point(message_clear_start:scan_context_io.RingKey)
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  data_.Clear();
  _internal_metadata_.Clear<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

const char* RingKey::_InternalParse(const char* ptr, ::PROTOBUF_NAMESPACE_ID::internal::ParseContext* ctx) {
#define CHK_(x) if (PROTOBUF_PREDICT_FALSE(!(x))) goto failure
  while (!ctx->Done(&ptr)) {
    ::PROTOBUF_NAMESPACE_ID::uint32 tag;
    ptr = ::PROTOBUF_NAMESPACE_ID::internal::ReadTag(ptr, &tag);
    CHK_(ptr);
    switch (tag >> 3) {
      // repeated float data = 1;
      case 1:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 13)) {
          ptr -= 1;
          do {
            ptr += 1;
            _internal_add_data(::PROTOBUF_NAMESPACE_ID::internal::UnalignedLoad<float>(ptr));
            ptr += sizeof(float);
            if (!ctx->DataAvailable(ptr)) break;
          } while (::PROTOBUF_NAMESPACE_ID::internal::ExpectTag<13>(ptr));
        } else if (static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 10) {
          ptr = ::PROTOBUF_NAMESPACE_ID::internal::PackedFloatParser(_internal_mutable_data(), ptr, ctx);
          CHK_(ptr);
        } else goto handle_unusual;
        continue;
      default: {
      handle_unusual:
        if ((tag & 7) == 4 || tag == 0) {
          ctx->SetLastTag(tag);
          goto success;
        }
        ptr = UnknownFieldParse(tag,
            _internal_metadata_.mutable_unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(),
            ptr, ctx);
        CHK_(ptr != nullptr);
        continue;
      }
    }  // switch
  }  // while
success:
  return ptr;
failure:
  ptr = nullptr;
  goto success;
#undef CHK_
}

::PROTOBUF_NAMESPACE_ID::uint8* RingKey::_InternalSerialize(
    ::PROTOBUF_NAMESPACE_ID::uint8* target, ::PROTOBUF_NAMESPACE_ID::io::EpsCopyOutputStream* stream) const {
  // @@protoc_insertion_point(serialize_to_array_start:scan_context_io.RingKey)
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  // repeated float data = 1;
  for (int i = 0, n = this->_internal_data_size(); i < n; i++) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteFloatToArray(1, this->_internal_data(i), target);
  }

  if (PROTOBUF_PREDICT_FALSE(_internal_metadata_.have_unknown_fields())) {
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormat::InternalSerializeUnknownFieldsToArray(
        _internal_metadata_.unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(::PROTOBUF_NAMESPACE_ID::UnknownFieldSet::default_instance), target, stream);
  }
  // @@protoc_insertion_point(serialize_to_array_end:scan_context_io.RingKey)
  return target;
}

size_t RingKey::ByteSizeLong() const {
// @@protoc_insertion_point(message_byte_size_start:scan_context_io.RingKey)
  size_t total_size = 0;

  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  // repeated float data = 1;
  {
    unsigned int count = static_cast<unsigned int>(this->_internal_data_size());
    size_t data_size = 4UL * count;
    total_size += 1 *
                  ::PROTOBUF_NAMESPACE_ID::internal::FromIntSize(this->_internal_data_size());
    total_size += data_size;
  }

  if (PROTOBUF_PREDICT_FALSE(_internal_metadata_.have_unknown_fields())) {
    return ::PROTOBUF_NAMESPACE_ID::internal::ComputeUnknownFieldsSize(
        _internal_metadata_, total_size, &_cached_size_);
  }
  int cached_size = ::PROTOBUF_NAMESPACE_ID::internal::ToCachedSize(total_size);
  SetCachedSize(cached_size);
  return total_size;
}

void RingKey::MergeFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) {
// @@protoc_insertion_point(generalized_merge_from_start:scan_context_io.RingKey)
  GOOGLE_DCHECK_NE(&from, this);
  const RingKey* source =
      ::PROTOBUF_NAMESPACE_ID::DynamicCastToGenerated<RingKey>(
          &from);
  if (source == nullptr) {
  // @@protoc_insertion_point(generalized_merge_from_cast_fail:scan_context_io.RingKey)
    ::PROTOBUF_NAMESPACE_ID::internal::ReflectionOps::Merge(from, this);
  } else {
  // @@protoc_insertion_point(generalized_merge_from_cast_success:scan_context_io.RingKey)
    MergeFrom(*source);
  }
}

void RingKey::MergeFrom(const RingKey& from) {
// @@protoc_insertion_point(class_specific_merge_from_start:scan_context_io.RingKey)
  GOOGLE_DCHECK_NE(&from, this);
  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  data_.MergeFrom(from.data_);
}

void RingKey::CopyFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) {
// @@protoc_insertion_point(generalized_copy_from_start:scan_context_io.RingKey)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

void RingKey::CopyFrom(const RingKey& from) {
// @@protoc_insertion_point(class_specific_copy_from_start:scan_context_io.RingKey)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool RingKey::IsInitialized() const {
  return true;
}

void RingKey::InternalSwap(RingKey* other) {
  using std::swap;
  _internal_metadata_.Swap<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(&other->_internal_metadata_);
  data_.InternalSwap(&other->data_);
}

::PROTOBUF_NAMESPACE_ID::Metadata RingKey::GetMetadata() const {
  return GetMetadataStatic();
}


// ===================================================================

class RingKeys::_Internal {
 public:
};

RingKeys::RingKeys(::PROTOBUF_NAMESPACE_ID::Arena* arena)
  : ::PROTOBUF_NAMESPACE_ID::Message(arena),
  data_(arena) {
  SharedCtor();
  RegisterArenaDtor(arena);
  // @@protoc_insertion_point(arena_constructor:scan_context_io.RingKeys)
}
RingKeys::RingKeys(const RingKeys& from)
  : ::PROTOBUF_NAMESPACE_ID::Message(),
      data_(from.data_) {
  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
  // @@protoc_insertion_point(copy_constructor:scan_context_io.RingKeys)
}

void RingKeys::SharedCtor() {
  ::PROTOBUF_NAMESPACE_ID::internal::InitSCC(&scc_info_RingKeys_ring_5fkeys_2eproto.base);
}

RingKeys::~RingKeys() {
  // @@protoc_insertion_point(destructor:scan_context_io.RingKeys)
  SharedDtor();
  _internal_metadata_.Delete<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

void RingKeys::SharedDtor() {
  GOOGLE_DCHECK(GetArena() == nullptr);
}

void RingKeys::ArenaDtor(void* object) {
  RingKeys* _this = reinterpret_cast< RingKeys* >(object);
  (void)_this;
}
void RingKeys::RegisterArenaDtor(::PROTOBUF_NAMESPACE_ID::Arena*) {
}
void RingKeys::SetCachedSize(int size) const {
  _cached_size_.Set(size);
}
const RingKeys& RingKeys::default_instance() {
  ::PROTOBUF_NAMESPACE_ID::internal::InitSCC(&::scc_info_RingKeys_ring_5fkeys_2eproto.base);
  return *internal_default_instance();
}


void RingKeys::Clear() {
// @@protoc_insertion_point(message_clear_start:scan_context_io.RingKeys)
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  data_.Clear();
  _internal_metadata_.Clear<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

const char* RingKeys::_InternalParse(const char* ptr, ::PROTOBUF_NAMESPACE_ID::internal::ParseContext* ctx) {
#define CHK_(x) if (PROTOBUF_PREDICT_FALSE(!(x))) goto failure
  while (!ctx->Done(&ptr)) {
    ::PROTOBUF_NAMESPACE_ID::uint32 tag;
    ptr = ::PROTOBUF_NAMESPACE_ID::internal::ReadTag(ptr, &tag);
    CHK_(ptr);
    switch (tag >> 3) {
      // repeated .scan_context_io.RingKey data = 1;
      case 1:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 10)) {
          ptr -= 1;
          do {
            ptr += 1;
            ptr = ctx->ParseMessage(_internal_add_data(), ptr);
            CHK_(ptr);
            if (!ctx->DataAvailable(ptr)) break;
          } while (::PROTOBUF_NAMESPACE_ID::internal::ExpectTag<10>(ptr));
        } else goto handle_unusual;
        continue;
      default: {
      handle_unusual:
        if ((tag & 7) == 4 || tag == 0) {
          ctx->SetLastTag(tag);
          goto success;
        }
        ptr = UnknownFieldParse(tag,
            _internal_metadata_.mutable_unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(),
            ptr, ctx);
        CHK_(ptr != nullptr);
        continue;
      }
    }  // switch
  }  // while
success:
  return ptr;
failure:
  ptr = nullptr;
  goto success;
#undef CHK_
}

::PROTOBUF_NAMESPACE_ID::uint8* RingKeys::_InternalSerialize(
    ::PROTOBUF_NAMESPACE_ID::uint8* target, ::PROTOBUF_NAMESPACE_ID::io::EpsCopyOutputStream* stream) const {
  // @@protoc_insertion_point(serialize_to_array_start:scan_context_io.RingKeys)
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  // repeated .scan_context_io.RingKey data = 1;
  for (unsigned int i = 0,
      n = static_cast<unsigned int>(this->_internal_data_size()); i < n; i++) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::
      InternalWriteMessage(1, this->_internal_data(i), target, stream);
  }

  if (PROTOBUF_PREDICT_FALSE(_internal_metadata_.have_unknown_fields())) {
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormat::InternalSerializeUnknownFieldsToArray(
        _internal_metadata_.unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(::PROTOBUF_NAMESPACE_ID::UnknownFieldSet::default_instance), target, stream);
  }
  // @@protoc_insertion_point(serialize_to_array_end:scan_context_io.RingKeys)
  return target;
}

size_t RingKeys::ByteSizeLong() const {
// @@protoc_insertion_point(message_byte_size_start:scan_context_io.RingKeys)
  size_t total_size = 0;

  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  // repeated .scan_context_io.RingKey data = 1;
  total_size += 1UL * this->_internal_data_size();
  for (const auto& msg : this->data_) {
    total_size +=
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::MessageSize(msg);
  }

  if (PROTOBUF_PREDICT_FALSE(_internal_metadata_.have_unknown_fields())) {
    return ::PROTOBUF_NAMESPACE_ID::internal::ComputeUnknownFieldsSize(
        _internal_metadata_, total_size, &_cached_size_);
  }
  int cached_size = ::PROTOBUF_NAMESPACE_ID::internal::ToCachedSize(total_size);
  SetCachedSize(cached_size);
  return total_size;
}

void RingKeys::MergeFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) {
// @@protoc_insertion_point(generalized_merge_from_start:scan_context_io.RingKeys)
  GOOGLE_DCHECK_NE(&from, this);
  const RingKeys* source =
      ::PROTOBUF_NAMESPACE_ID::DynamicCastToGenerated<RingKeys>(
          &from);
  if (source == nullptr) {
  // @@protoc_insertion_point(generalized_merge_from_cast_fail:scan_context_io.RingKeys)
    ::PROTOBUF_NAMESPACE_ID::internal::ReflectionOps::Merge(from, this);
  } else {
  // @@protoc_insertion_point(generalized_merge_from_cast_success:scan_context_io.RingKeys)
    MergeFrom(*source);
  }
}

void RingKeys::MergeFrom(const RingKeys& from) {
// @@protoc_insertion_point(class_specific_merge_from_start:scan_context_io.RingKeys)
  GOOGLE_DCHECK_NE(&from, this);
  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  data_.MergeFrom(from.data_);
}

void RingKeys::CopyFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) {
// @@protoc_insertion_point(generalized_copy_from_start:scan_context_io.RingKeys)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

void RingKeys::CopyFrom(const RingKeys& from) {
// @@protoc_insertion_point(class_specific_copy_from_start:scan_context_io.RingKeys)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool RingKeys::IsInitialized() const {
  return true;
}

void RingKeys::InternalSwap(RingKeys* other) {
  using std::swap;
  _internal_metadata_.Swap<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(&other->_internal_metadata_);
  data_.InternalSwap(&other->data_);
}

::PROTOBUF_NAMESPACE_ID::Metadata RingKeys::GetMetadata() const {
  return GetMetadataStatic();
}


// @@protoc_insertion_point(namespace_scope)
}  // namespace scan_context_io
PROTOBUF_NAMESPACE_OPEN
template<> PROTOBUF_NOINLINE ::scan_context_io::RingKey* Arena::CreateMaybeMessage< ::scan_context_io::RingKey >(Arena* arena) {
  return Arena::CreateMessageInternal< ::scan_context_io::RingKey >(arena);
}
template<> PROTOBUF_NOINLINE ::scan_context_io::RingKeys* Arena::CreateMaybeMessage< ::scan_context_io::RingKeys >(Arena* arena) {
  return Arena::CreateMessageInternal< ::scan_context_io::RingKeys >(arena);
}
PROTOBUF_NAMESPACE_CLOSE

// @@protoc_insertion_point(global_scope)
#include <google/protobuf/port_undef.inc>