# CRITICAL: Protobuf compatibility patch MUST happen first
# This allows TensorFlow to import successfully despite protobuf version mismatch
try:
    import google.protobuf
    from google.protobuf import _message
    
    # Ensure ValidateProtobufRuntimeVersion exists at module level
    if not hasattr(_message, 'ValidateProtobufRuntimeVersion'):
        def ValidateProtobufRuntimeVersion(domain, version_string):
            """Validation stub to allow TensorFlow imports"""
            pass
        _message.ValidateProtobufRuntimeVersion = ValidateProtobufRuntimeVersion
        
except Exception as e:
    pass
