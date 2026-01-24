## Uruchomienie

1. Generator logow (tworzy `logs.csv` w katalogu glownym projektu):

```bash
dotnet run --project SyntheticLogGenerator/SyntheticLogGenerator.csproj -- SyntheticLogGenerator/generator.json
```

2. Konwersja CSV do Parquet z rozbiciem `AttributesJson`:

```bash
python LogAnalysis/load_and_parquet.py
```
