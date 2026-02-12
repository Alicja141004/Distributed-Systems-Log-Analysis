## Uruchomienie

1. Generator logow (tworzy `logs.csv` w katalogu glownym projektu):

```bash
dotnet run --project SyntheticLogGenerator/SyntheticLogGenerator.csproj -- SyntheticLogGenerator/generator.json
```

2. Konwersja CSV do Parquet z rozbiciem `AttributesJson`:

```bash
python LogAnalysis/load_and_parquet.py
```

3. Uruchomienie pipeline zadań 5.1-5.9 + gotowy prompt do AI:

```bash
cd LogAnalysis
sh run_tasks.sh
```
