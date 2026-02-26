# Changelog

## [0.1.0](https://github.com/seanreed1111/icd10-codes/releases/tag/0.1.0) - 2026-02-26

### Added

- Add CodiEsp diagnosis parser with code normalization — uppercases all letters and strips punctuation before aggregating codes per clinical text file ([`9e0781e`](https://github.com/seanreed1111/icd10-codes/commit/9e0781eebb25678f449e7fcab6037c4636c64d91), [#1](https://github.com/seanreed1111/icd10-codes/pull/1))
- Add CLI for ICD-10 code lookup by natural-language condition description ([`7eccfd4`](https://github.com/seanreed1111/icd10-codes/commit/7eccfd45fd158bb5ce64d7015b8ec0de7ec4495f))
- Add natural-language search functions to KnowledgeBase ([`a4afeb4`](https://github.com/seanreed1111/icd10-codes/commit/a4afeb473b303a085e734ddeceb822c03656a8a1))
- Add load/save methods to KnowledgeBase for Parquet persistence ([`3debd23`](https://github.com/seanreed1111/icd10-codes/commit/3debd23e2b711e280315f72de8e63c61ef08f2b4))
- Add TF-IDF search examples ([`0e7da8a`](https://github.com/seanreed1111/icd10-codes/commit/0e7da8aaa6512a98a1a7a6244dcaba6a5a6e7e68))
- Add CodiEsp test datasets ([`a8f7b88`](https://github.com/seanreed1111/icd10-codes/commit/a8f7b88e82b538a3df625eb4559ffd112108b576))
- Add pre-commit hooks with ruff linting and formatting ([`152bcf7`](https://github.com/seanreed1111/icd10-codes/commit/152bcf756c49e295b56656344596d31df746fbf2))
- Add initial ICD-10 codes CSV ([`c4cad6b`](https://github.com/seanreed1111/icd10-codes/commit/c4cad6be9e23491651d4c64e9f5b14dfaae225cd))

### Changed

- Restructure project directories ([`92e90e2`](https://github.com/seanreed1111/icd10-codes/commit/92e90e2e45719fabbfccbec55e8e465b0bd65968))
- Renormalize code data ([`afcffcd`](https://github.com/seanreed1111/icd10-codes/commit/afcffcd96224a12b429916e8f0da7f4a34ee9931))

### Fixed

- Fix tests to use `load_from_parquet` after persistence refactor ([`74168ae`](https://github.com/seanreed1111/icd10-codes/commit/74168aecda8f680caf4a1e4434aa5f758043dd60))
