class ArtifactError(RuntimeError):
    pass

class SessionNotFound(KeyError):
    pass

class ProviderUnavailable(RuntimeError):
    pass
