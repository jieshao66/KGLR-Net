# Security guidance

KGLR-Net loads model checkpoints and preprocessed arrays from local storage.
Treat both as executable or structured inputs and use only files that you
created or obtained from an official, trusted source.

- Do not load checkpoints received through untrusted links.
- Verify downloaded checkpoint hashes when the publisher supplies one.
- Keep datasets, credentials, and machine-specific paths outside the Git
  repository.
- Report suspected vulnerabilities privately to the corresponding authors
  listed in `CITATION.cff`.

The public programs use PyTorch's restricted weight loader when the installed
PyTorch version supports it. Preprocessed samples use compressed `.npz`
archives and are loaded with NumPy pickling disabled.
