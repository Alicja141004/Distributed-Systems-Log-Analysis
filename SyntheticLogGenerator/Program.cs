using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json;

namespace SyntheticLogGenerator
{
    /// <summary>
    /// Poziom logu.
    /// </summary>
    public enum LogPriority
    {
        Info,
        Warn,
        Err,
        Crit
    }

    /// <summary>
    /// Typ zdarzenia (można łatwo rozszerzyć).
    /// </summary>
    public enum EventKind
    {
        BusinessAction,
        TechnicalMetric,
        Heartbeat,
        Audit,
        Error,
        Trace
    }

    /// <summary>
    /// Konfiguracja generatora – parametryzowana z pliku JSON.
    /// </summary>
    public class GenerationConfig
    {
        public string OutputPath { get; set; } = "logs.csv";
        public double TargetSizeGb { get; set; } = 1.0;
        public int Seed { get; set; } = 123;
        public int SystemCount { get; set; } = 5;

        // Parametry zjawisk
        public int ExpectedSpikeWindows { get; set; } = 10;
        public int ExpectedFailureWindows { get; set; } = 5;
        public int ExpectedTrendWindows { get; set; } = 5;
        public double AnomalyRate { get; set; } = 0.0005; // 0.05%

        // Uśrednione parametry „czasowe” – w sekundach
        public double AverageInterEventTimeSeconds { get; set; } = 0.5;

        // Wygodny wrapper
        public TimeSpan AverageInterEventTime =>
            TimeSpan.FromSeconds(AverageInterEventTimeSeconds);

        public long TargetSizeBytes =>
            (long)(TargetSizeGb * 1024d * 1024d * 1024d);

        public static GenerationConfig LoadFromFile(string path)
        {
            if (!File.Exists(path))
            {
                throw new FileNotFoundException(
                    $"Nie znaleziono pliku konfiguracyjnego: {path}");
            }

            var json = File.ReadAllText(path);
            var options = new JsonSerializerOptions
            {
                PropertyNameCaseInsensitive = true
            };

            var cfg = JsonSerializer.Deserialize<GenerationConfig>(json, options)
                      ?? new GenerationConfig();

            // Minimalna walidacja / ograniczenia
            cfg.TargetSizeGb = Math.Clamp(cfg.TargetSizeGb, 1.0, 100.0);
            cfg.SystemCount = Math.Max(1, cfg.SystemCount);
            cfg.ExpectedSpikeWindows = Math.Max(0, cfg.ExpectedSpikeWindows);
            cfg.ExpectedFailureWindows = Math.Max(0, cfg.ExpectedFailureWindows);
            cfg.ExpectedTrendWindows = Math.Max(0, cfg.ExpectedTrendWindows);
            cfg.AnomalyRate = Math.Max(0.0, cfg.AnomalyRate);
            if (cfg.AverageInterEventTimeSeconds <= 0)
                cfg.AverageInterEventTimeSeconds = 0.5;

            return cfg;
        }
    }

    /// <summary>
    /// Jeden wpis logu / zdarzenia.
    /// </summary>
    public class LogEntry
    {
        public DateTime Timestamp { get; set; }
        public LogPriority Priority { get; set; }
        public string User { get; set; } = "";
        public string SourceSystem { get; set; } = "";
        public EventKind EventKind { get; set; }
        public int EventCode { get; set; }
        public string TransactionId { get; set; } = "";
        public string CorrelationId { get; set; } = "";
        public string Description { get; set; } = "";
        public bool IsAnomaly { get; set; }
        public string Scenario { get; set; } = "";
        public int LatencyMs { get; set; }
        public Dictionary<string, object> Attributes { get; set; } = new();

        public string ToCsvLine()
        {
            string attrsJson = JsonSerializer.Serialize(Attributes);

            return string.Join(";",
                Timestamp.ToString("o"),
                Priority.ToString().ToLowerInvariant(),
                User,
                SourceSystem,
                EventKind.ToString(),
                EventCode.ToString(CultureInfo.InvariantCulture),
                TransactionId,
                CorrelationId,
                Description.Replace(";", ","),
                LatencyMs.ToString(CultureInfo.InvariantCulture),
                IsAnomaly ? "1" : "0",
                Scenario,
                attrsJson.Replace(";", ",")
            );
        }

        public static string CsvHeader =>
            "Timestamp;Priority;User;SourceSystem;EventKind;EventCode;TransactionId;CorrelationId;Description;LatencyMs;IsAnomaly;Scenario;AttributesJson";
    }

    /// <summary>
    /// Liczniki zjawisk – żebyś mógł/mogła studentom powiedzieć, ile czego jest.
    /// </summary>
    public class PhenomenaCounters
    {
        public long TotalEvents;
        public long SpikeEvents;
        public long FailureEvents;
        public long TrendEvents;
        public long AnomalyEvents;

        public long SpikeByLatencyEvents;
        public long SpikeByCpuEvents;
        public long SpikeByQpsEvents;
    }

    /// <summary>
    /// Dane do pliku statystyk *.stats.json.
    /// </summary>
    public class GenerationStats
    {
        public DateTime GeneratedAtUtc { get; set; }
        public string OutputPath { get; set; } = "";
        public double TargetSizeGb { get; set; }
        public long TargetSizeBytes { get; set; }
        public long ActualSizeBytes { get; set; }

        public long TotalEvents { get; set; }
        public long SpikeEvents { get; set; }
        public long FailureEvents { get; set; }
        public long TrendEvents { get; set; }
        public long AnomalyEvents { get; set; }

        public long SpikeByLatencyEvents { get; set; }
        public long SpikeByCpuEvents { get; set; }
        public long SpikeByQpsEvents { get; set; }

        public int ExpectedSpikeWindows { get; set; }
        public int ExpectedFailureWindows { get; set; }
        public int ExpectedTrendWindows { get; set; }
        public double AnomalyRate { get; set; }
        public double AverageInterEventTimeSeconds { get; set; }

        public double SpikeLatencyThresholdMs { get; set; }
        public double SpikeCpuThresholdPercent { get; set; }
        public double SpikeQpsThreshold { get; set; }

        public int SystemCount { get; set; }
        public int Seed { get; set; }
    }

    public static class Program
    {
        // Domyślne nazwy systemów – co najmniej 5 powiązanych.
        private static readonly string[] DefaultSystems =
        {
            "AuthService",
            "ApiGateway",
            "OrderService",
            "PaymentService",
            "NotificationService",
            "InventoryService",
            "ReportingService"
        };

        private static readonly string[] SampleUsers =
        {
            "alice", "bob", "carol", "dave", "eve",
            "student01", "student02", "qa_user", "service_account"
        };

        private static readonly string[] BusinessScenarios =
        {
            "OrderPlacement",
            "UserLogin",
            "PasswordReset",
            "CheckoutWithCard",
            "CheckoutWithTransfer",
            "StockReservation",
            "EmailConfirmation",
            "RefundProcessing",
            "BulkImport",
            "ReportGeneration"
        };

        // PROGI SPIKE'ÓW (spójne z BuildMetrics)
        private const int SpikeLatencyThresholdMs = 300;
        private const double SpikeCpuThresholdPercent = 80.0;
        private const double SpikeQpsThreshold = 75.0;

        public static void Main(string[] args)
        {
            Console.OutputEncoding = Encoding.UTF8;

            string configPath = args.Length > 0
                ? args[0]
                : "generator.json";

            GenerationConfig config;
            try
            {
                config = GenerationConfig.LoadFromFile(configPath);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Błąd ładowania konfiguracji z pliku '{configPath}':");
                Console.WriteLine(ex.Message);
                return;
            }

            Console.WriteLine("=== Synthetic Log Generator ===");
            Console.WriteLine($"Plik konfiguracyjny : {configPath}");
            Console.WriteLine($"Plik wyjściowy     : {config.OutputPath}");
            Console.WriteLine($"Rozmiar docelowy   : {config.TargetSizeGb} GB (~{config.TargetSizeBytes:N0} bajtów)");
            Console.WriteLine($"Systemów           : {config.SystemCount}");
            Console.WriteLine($"Spike windows      : {config.ExpectedSpikeWindows}");
            Console.WriteLine($"Failure windows    : {config.ExpectedFailureWindows}");
            Console.WriteLine($"Trend windows      : {config.ExpectedTrendWindows}");
            Console.WriteLine($"Anomaly rate       : {config.AnomalyRate}");
            Console.WriteLine($"Śr. odstęp zdarzeń : {config.AverageInterEventTimeSeconds} s");
            Console.WriteLine();

            var counters = new PhenomenaCounters();
            long actualSizeBytes = GenerateFile(config, counters);

            Console.WriteLine();
            Console.WriteLine("=== Statystyki (przybliżone) ===");
            Console.WriteLine($"Liczba zdarzeń           : {counters.TotalEvents:N0}");
            Console.WriteLine($"Spike events (eventCode) : {counters.SpikeEvents:N0}");
            Console.WriteLine($"Failure events           : {counters.FailureEvents:N0}");
            Console.WriteLine($"Trend events             : {counters.TrendEvents:N0}");
            Console.WriteLine($"Anomaly events           : {counters.AnomalyEvents:N0}");
            Console.WriteLine($"Spike by Latency (>= {SpikeLatencyThresholdMs} ms): {counters.SpikeByLatencyEvents:N0}");
            Console.WriteLine($"Spike by CPU (>= {SpikeCpuThresholdPercent}%):     {counters.SpikeByCpuEvents:N0}");
            Console.WriteLine($"Spike by QPS (>= {SpikeQpsThreshold}):             {counters.SpikeByQpsEvents:N0}");

            SaveStats(config, counters, actualSizeBytes);

            Console.WriteLine("Gotowe.");
        }

        private static long GenerateFile(GenerationConfig cfg, PhenomenaCounters counters)
        {
            var rnd = new Random(cfg.Seed);

            DateTime start = DateTime.UtcNow.AddDays(-30);
            DateTime currentTime = start;

            var encoding = new UTF8Encoding(false);
            long bytesWritten = 0;

            var systems = DefaultSystems.Take(cfg.SystemCount)
                .Concat(
                    Enumerable.Range(0, Math.Max(0, cfg.SystemCount - DefaultSystems.Length))
                              .Select(i => $"ExtraSystem{i + 1}")
                )
                .ToArray();

            long approxEventsForSize = (long)(cfg.TargetSizeBytes / 300.0); // ~300 bajtów na wpis

            var spikeWindows = CreateWindows(rnd, cfg.ExpectedSpikeWindows, approxEventsForSize, 0.005);    // 0.5% długości
            var failureWindows = CreateWindows(rnd, cfg.ExpectedFailureWindows, approxEventsForSize, 0.002); // 0.2%
            var trendWindows = CreateWindows(rnd, cfg.ExpectedTrendWindows, approxEventsForSize, 0.02);    // 2%

            int baseLatency = 50; // ms

            using var fs = new FileStream(cfg.OutputPath, FileMode.Create, FileAccess.Write, FileShare.Read, 1 << 20);
            using var writer = new StreamWriter(fs, encoding, 1 << 20);

            void WriteLineAndCount(string line)
            {
                writer.WriteLine(line);
                bytesWritten += encoding.GetByteCount(line) + encoding.GetByteCount(Environment.NewLine);
            }

            WriteLineAndCount(LogEntry.CsvHeader);

            long eventIndex = 0;
            long lastReportBytes = 0;

            while (bytesWritten < cfg.TargetSizeBytes)
            {
                GenerateScenarioTransaction(
                    cfg,
                    systems,
                    rnd,
                    spikeWindows,
                    failureWindows,
                    trendWindows,
                    ref currentTime,
                    ref eventIndex,
                    baseLatency,
                    counters,
                    WriteLineAndCount
                );

                // Dodatkowe pojedyncze zdarzenia techniczne
                if (rnd.NextDouble() < 0.2)
                {
                    var single = GenerateSingleTechnicalEvent(
                        cfg,
                        systems,
                        rnd,
                        spikeWindows,
                        failureWindows,
                        trendWindows,
                        ref currentTime,
                        eventIndex,
                        baseLatency,
                        counters
                    );

                    WriteLineAndCount(single.ToCsvLine());
                    eventIndex++;
                }

                if (bytesWritten - lastReportBytes > 200 * 1024 * 1024)
                {
                    lastReportBytes = bytesWritten;
                    double progress = (double)bytesWritten / cfg.TargetSizeBytes * 100.0;
                    Console.WriteLine($"Postęp: {progress:F1}% ({bytesWritten / (1024.0 * 1024.0 * 1024.0):F2} GB)");
                }
            }

            writer.Flush();
            return bytesWritten;
        }

        /// <summary>
        /// Zapisuje plik statystyk *.stats.json.
        /// </summary>
        private static void SaveStats(GenerationConfig cfg, PhenomenaCounters counters, long actualSizeBytes)
        {
            var stats = new GenerationStats
            {
                GeneratedAtUtc = DateTime.UtcNow,
                OutputPath = cfg.OutputPath,
                TargetSizeGb = cfg.TargetSizeGb,
                TargetSizeBytes = cfg.TargetSizeBytes,
                ActualSizeBytes = actualSizeBytes,

                TotalEvents = counters.TotalEvents,
                SpikeEvents = counters.SpikeEvents,
                FailureEvents = counters.FailureEvents,
                TrendEvents = counters.TrendEvents,
                AnomalyEvents = counters.AnomalyEvents,

                SpikeByLatencyEvents = counters.SpikeByLatencyEvents,
                SpikeByCpuEvents = counters.SpikeByCpuEvents,
                SpikeByQpsEvents = counters.SpikeByQpsEvents,

                ExpectedSpikeWindows = cfg.ExpectedSpikeWindows,
                ExpectedFailureWindows = cfg.ExpectedFailureWindows,
                ExpectedTrendWindows = cfg.ExpectedTrendWindows,
                AnomalyRate = cfg.AnomalyRate,
                AverageInterEventTimeSeconds = cfg.AverageInterEventTimeSeconds,

                SpikeLatencyThresholdMs = SpikeLatencyThresholdMs,
                SpikeCpuThresholdPercent = SpikeCpuThresholdPercent,
                SpikeQpsThreshold = SpikeQpsThreshold,

                SystemCount = cfg.SystemCount,
                Seed = cfg.Seed
            };

            string statsPath = cfg.OutputPath + ".stats.json";

            var options = new JsonSerializerOptions
            {
                WriteIndented = true
            };

            File.WriteAllText(statsPath, JsonSerializer.Serialize(stats, options));
            Console.WriteLine($"Zapisano statystyki do pliku: {statsPath}");
        }

        /// <summary>
        /// Tworzy losowe okna (start, end) po indeksie zdarzenia.
        /// </summary>
        private static List<(long Start, long End)> CreateWindows(Random rnd, int count, long totalEvents, double windowFraction)
        {
            var result = new List<(long Start, long End)>();
            if (count <= 0 || totalEvents <= 0 || windowFraction <= 0)
                return result;

            long windowLength = Math.Max(10, (long)(totalEvents * windowFraction));

            for (int i = 0; i < count; i++)
            {
                long center = (long)(rnd.NextDouble() * totalEvents);
                long start = center - windowLength / 2;
                if (start < 0) start = 0;
                long end = start + windowLength;
                if (end >= totalEvents) end = totalEvents - 1;
                if (end < start) continue;

                result.Add((start, end));
            }

            result.Sort((a, b) => a.Start.CompareTo(b.Start));
            return result;
        }

        private static bool IsInAnyWindow(List<(long Start, long End)> windows, long index)
        {
            foreach (var w in windows)
            {
                if (index >= w.Start && index <= w.End)
                    return true;
            }
            return false;
        }

        /// <summary>
        /// Buduje wartości metryk numerycznych tak, aby spike/trend/failure/anomaly
        /// były realnie widoczne w danych.
        /// </summary>
        private static void BuildMetrics(
            Random rnd,
            long eventIndex,
            int baseLatency,
            bool inSpike,
            bool inFailure,
            bool inTrend,
            bool isAnomaly,
            bool isBackgroundMetric,
            out int latencyMs,
            out double cpuUsage,
            out int memoryUsageMb,
            out int requestSizeBytes,
            out int responseSizeBytes,
            out int diskQueue,
            out int networkErrors,
            out int retries,
            out double localQps,
            out bool isSpikeByLatency,
            out bool isSpikeByCpu,
            out bool isSpikeByQps)
        {
            // 1. Latency
            int baselineLatency = isBackgroundMetric
                ? baseLatency / 2 + rnd.Next(-5, 15)
                : baseLatency + rnd.Next(-10, 20);

            if (baselineLatency < 1) baselineLatency = 1;

            double trendFactor = 1.0;
            if (inTrend)
            {
                // rosnący trend wraz z eventIndex (łagodny)
                trendFactor += (eventIndex % 5000) / 5000.0 * 1.5; // max ~2.5x
            }

            int latency = (int)(baselineLatency * trendFactor);

            if (inSpike)
            {
                // spike – dodatkowy boost
                latency += rnd.Next(150, 400); // 150–400ms
            }

            if (inFailure)
            {
                // failure – często jeszcze większe opóźnienie
                latency += rnd.Next(500, 2000);
            }

            if (isAnomaly)
            {
                // anomaly – duże, raw wartości
                latency = rnd.Next(2000, 10000);
            }

            if (latency < 1) latency = 1;

            latencyMs = latency;

            // 2. CPU usage
            double cpu = 10 + rnd.NextDouble() * 45; // 10–55%

            if (inTrend)
            {
                cpu += rnd.Next(0, 15); // lekki wzrost w trendzie
            }

            if (inSpike)
            {
                cpu = Math.Max(cpu, 80 + rnd.NextDouble() * 20); // 80–100%
            }

            if (inFailure)
            {
                cpu = Math.Max(cpu, 60 + rnd.NextDouble() * 35); // 60–95%
            }

            if (isAnomaly)
            {
                // anomaly – CPU bardzo niskie albo prawie 100
                cpu = rnd.NextDouble() < 0.5
                    ? rnd.Next(0, 5) + rnd.NextDouble()  // ~0–6%
                    : 98 + rnd.NextDouble() * 2;         // 98–100%
            }

            if (cpu < 0) cpu = 0;
            if (cpu > 100) cpu = 100;
            cpuUsage = cpu;

            // 3. Memory usage
            int mem = 200 + rnd.Next(0, 2800); // 200–3000 MB
            if (inTrend)
            {
                mem += rnd.Next(0, 500);
            }

            if (inSpike)
            {
                mem += rnd.Next(200, 800);
            }

            if (inFailure)
            {
                mem += rnd.Next(0, 500);
            }

            if (isAnomaly)
            {
                mem = Math.Max(50, mem + rnd.Next(-500, 2000));
            }

            memoryUsageMb = mem;

            // 4. Request / response size
            int req = rnd.Next(300, 20000);
            int resp = rnd.Next(300, 20000);

            if (inSpike)
            {
                req *= rnd.Next(2, 4);
                resp *= rnd.Next(2, 4);
            }

            if (isAnomaly && rnd.NextDouble() < 0.5)
            {
                // dziwnie małe albo duże
                if (rnd.NextDouble() < 0.5)
                {
                    req = rnd.Next(1, 100);
                    resp = rnd.Next(1, 100);
                }
                else
                {
                    req = rnd.Next(50000, 200000);
                    resp = rnd.Next(50000, 200000);
                }
            }

            requestSizeBytes = req;
            responseSizeBytes = resp;

            // 5. Disk queue / network errors
            int dq = rnd.Next(0, 20);
            int ne = rnd.Next(0, 3);

            if (inTrend)
            {
                dq += rnd.Next(0, 10);
            }

            if (inSpike)
            {
                dq += rnd.Next(20, 100);
                ne += rnd.Next(3, 15);
            }

            if (inFailure)
            {
                dq += rnd.Next(50, 200);
                ne += rnd.Next(5, 30);
            }

            if (isAnomaly)
            {
                dq += rnd.Next(0, 300);
                ne += rnd.Next(0, 50);
            }

            if (dq < 0) dq = 0;
            if (ne < 0) ne = 0;
            diskQueue = dq;
            networkErrors = ne;

            // 6. Local QPS (syntetyczne)
            double qps = 10 + rnd.NextDouble() * 40; // normalne 10–50

            if (inSpike)
            {
                qps = 100 + rnd.NextDouble() * 400; // spike 100–500
            }

            if (isAnomaly && rnd.NextDouble() < 0.3)
            {
                // dziwnie niskie / wysokie
                qps = rnd.NextDouble() < 0.5
                    ? 0.1 + rnd.NextDouble()   // prawie brak ruchu
                    : 800 + rnd.NextDouble() * 400; // ogromny ruch
            }

            localQps = qps;

            // 7. Retries
            int r = 0;
            if (inFailure)
            {
                r = rnd.Next(1, 5);
            }
            else if (isAnomaly && rnd.NextDouble() < 0.3)
            {
                r = rnd.Next(1, 3);
            }

            retries = r;

            // 8. Spike detekcja "prawdziwa" wg metryk
            isSpikeByLatency = latencyMs >= SpikeLatencyThresholdMs;
            isSpikeByCpu = cpuUsage >= SpikeCpuThresholdPercent;
            isSpikeByQps = localQps >= SpikeQpsThreshold;
        }

        /// <summary>
        /// Generuje scenariusz transakcji rozłożony na kilka systemów.
        /// </summary>
        private static void GenerateScenarioTransaction(
            GenerationConfig cfg,
            string[] systems,
            Random rnd,
            List<(long Start, long End)> spikeWindows,
            List<(long Start, long End)> failureWindows,
            List<(long Start, long End)> trendWindows,
            ref DateTime currentTime,
            ref long eventIndex,
            int baseLatency,
            PhenomenaCounters counters,
            Action<string> writeLine)
        {
            string scenarioName = BusinessScenarios[rnd.Next(BusinessScenarios.Length)];
            string correlationId = Guid.NewGuid().ToString("N");
            string transactionId = Guid.NewGuid().ToString("N");
            string user = SampleUsers[rnd.Next(SampleUsers.Length)];

            int pipelineLength = Math.Min(5, systems.Length);
            var pipeline = systems
                .OrderBy(_ => rnd.Next())
                .Take(pipelineLength)
                .ToArray();

            int stepIndex = 0;

            foreach (var system in pipeline)
            {
                bool hasTwoSteps = rnd.NextDouble() < 0.6;
                int stepsInSystem = hasTwoSteps ? 2 : 1;

                for (int localStep = 0; localStep < stepsInSystem; localStep++)
                {
                    bool inSpike = IsInAnyWindow(spikeWindows, eventIndex);
                    bool inFailure = IsInAnyWindow(failureWindows, eventIndex);
                    bool inTrend = IsInAnyWindow(trendWindows, eventIndex);

                    var priority = LogPriority.Info;
                    var kind = EventKind.BusinessAction;
                    int eventCode = 200;
                    string description = $"{scenarioName}: step={stepIndex} system={system} status=OK";

                    bool isAnomaly = rnd.NextDouble() < cfg.AnomalyRate;

                    if (inFailure && rnd.NextDouble() < 0.4)
                    {
                        priority = LogPriority.Crit;
                        kind = EventKind.Error;
                        eventCode = 500;
                        description = $"{scenarioName}: step={stepIndex} system={system} FAILURE";
                        counters.FailureEvents++;
                    }
                    else if (inSpike && rnd.NextDouble() < 0.3)
                    {
                        priority = LogPriority.Warn;
                        kind = EventKind.Trace;
                        eventCode = 429; // zbyt duży ruch
                        description = $"{scenarioName}: step={stepIndex} system={system} spike load";
                        counters.SpikeEvents++;
                    }

                    if (inTrend)
                    {
                        counters.TrendEvents++;
                    }

                    if (isAnomaly)
                    {
                        priority = LogPriority.Err;
                        kind = EventKind.Error;
                        eventCode = 999; // nietypowy kod
                        description = $"{scenarioName}: step={stepIndex} system={system} ANOMALY";
                        counters.AnomalyEvents++;
                    }

                    BuildMetrics(
                        rnd,
                        eventIndex,
                        baseLatency,
                        inSpike,
                        inFailure,
                        inTrend,
                        isAnomaly,
                        isBackgroundMetric: false,
                        out int latencyMs,
                        out double cpuUsage,
                        out int memUsage,
                        out int reqSize,
                        out int respSize,
                        out int diskQueue,
                        out int netErrors,
                        out int retries,
                        out double localQps,
                        out bool isSpikeByLatency,
                        out bool isSpikeByCpu,
                        out bool isSpikeByQps
                    );

                    // zliczanie spike'ów rozpoznanych po metrykach
                    if (isSpikeByLatency) counters.SpikeByLatencyEvents++;
                    if (isSpikeByCpu) counters.SpikeByCpuEvents++;
                    if (isSpikeByQps) counters.SpikeByQpsEvents++;

                    var entry = new LogEntry
                    {
                        Timestamp = currentTime,
                        Priority = priority,
                        User = user,
                        SourceSystem = system,
                        EventKind = kind,
                        EventCode = eventCode,
                        TransactionId = transactionId,
                        CorrelationId = correlationId,
                        Description = description,
                        IsAnomaly = isAnomaly,
                        Scenario = scenarioName,
                        LatencyMs = latencyMs,
                        Attributes = new Dictionary<string, object>
                        {
                            ["ScenarioStepIndex"] = stepIndex,
                            ["ScenarioLocalStep"] = localStep,
                            ["SpikeWindow"] = inSpike,
                            ["FailureWindow"] = inFailure,
                            ["TrendWindow"] = inTrend,
                            ["CpuUsage"] = Math.Round(cpuUsage, 2),
                            ["MemoryUsageMb"] = memUsage,
                            ["RequestSizeBytes"] = reqSize,
                            ["ResponseSizeBytes"] = respSize,
                            ["Retries"] = retries,
                            ["SystemRole"] = InferSystemRole(system),
                            ["ClusterNode"] = $"node-{rnd.Next(1, 10)}",
                            ["Region"] = rnd.NextDouble() < 0.5 ? "eu-central" : "us-east",
                            ["BusinessKey"] = $"ORD-{rnd.Next(100000, 999999)}",
                            ["DiskQueueLength"] = diskQueue,
                            ["NetworkErrors"] = netErrors,
                            ["LocalQps"] = Math.Round(localQps, 2),
                            ["IsSpikeByLatency"] = isSpikeByLatency,
                            ["IsSpikeByCpu"] = isSpikeByCpu,
                            ["IsSpikeByQps"] = isSpikeByQps
                        }
                    };

                    writeLine(entry.ToCsvLine());
                    counters.TotalEvents++;
                    eventIndex++;

                    double jitterFactor = 0.3 + rnd.NextDouble() * 1.7; // 0.3–2.0
                    currentTime = currentTime.AddSeconds(
                        cfg.AverageInterEventTimeSeconds * jitterFactor);

                    stepIndex++;
                }
            }
        }

        private static string InferSystemRole(string systemName)
        {
            systemName = systemName.ToLowerInvariant();
            if (systemName.Contains("auth")) return "Auth";
            if (systemName.Contains("gateway")) return "Gateway";
            if (systemName.Contains("order")) return "Order";
            if (systemName.Contains("payment")) return "Payment";
            if (systemName.Contains("notification")) return "Notification";
            if (systemName.Contains("inventory")) return "Inventory";
            if (systemName.Contains("report")) return "Reporting";
            return "Generic";
        }

        /// <summary>
        /// Pojedyncze zdarzenie techniczne (heartbeat / metric).
        /// </summary>
        private static LogEntry GenerateSingleTechnicalEvent(
            GenerationConfig cfg,
            string[] systems,
            Random rnd,
            List<(long Start, long End)> spikeWindows,
            List<(long Start, long End)> failureWindows,
            List<(long Start, long End)> trendWindows,
            ref DateTime currentTime,
            long eventIndex,
            int baseLatency,
            PhenomenaCounters counters)
        {
            string system = systems[rnd.Next(systems.Length)];
            bool inSpike = IsInAnyWindow(spikeWindows, eventIndex);
            bool inFailure = IsInAnyWindow(failureWindows, eventIndex);
            bool inTrend = IsInAnyWindow(trendWindows, eventIndex);

            var kind = EventKind.TechnicalMetric;
            var priority = LogPriority.Info;
            int eventCode = 100;
            string description = $"Tech metric from {system}";

            bool isAnomaly = rnd.NextDouble() < cfg.AnomalyRate;

            if (inFailure && rnd.NextDouble() < 0.3)
            {
                kind = EventKind.Error;
                priority = LogPriority.Err;
                eventCode = 501;
                description = $"System {system} heartbeat failure";
                counters.FailureEvents++;
            }
            else if (inSpike && rnd.NextDouble() < 0.3)
            {
                priority = LogPriority.Warn;
                eventCode = 430;
                description = $"System {system} high QPS spike";
                counters.SpikeEvents++;
            }

            if (inTrend)
            {
                counters.TrendEvents++;
            }

            if (isAnomaly)
            {
                priority = LogPriority.Err;
                kind = EventKind.Error;
                eventCode = 998;
                description = $"System {system} ANOMALOUS metric";
                counters.AnomalyEvents++;
            }

            BuildMetrics(
                rnd,
                eventIndex,
                baseLatency,
                inSpike,
                inFailure,
                inTrend,
                isAnomaly,
                isBackgroundMetric: true,
                out int latencyMs,
                out double cpuUsage,
                out int memUsage,
                out int reqSize,
                out int respSize,
                out int diskQueue,
                out int netErrors,
                out int retries,
                out double localQps,
                out bool isSpikeByLatency,
                out bool isSpikeByCpu,
                out bool isSpikeByQps
            );

            if (isSpikeByLatency) counters.SpikeByLatencyEvents++;
            if (isSpikeByCpu) counters.SpikeByCpuEvents++;
            if (isSpikeByQps) counters.SpikeByQpsEvents++;

            string txId = Guid.NewGuid().ToString("N");
            string corrId = Guid.NewGuid().ToString("N");

            var entry = new LogEntry
            {
                Timestamp = currentTime,
                Priority = priority,
                User = "system",
                SourceSystem = system,
                EventKind = kind,
                EventCode = eventCode,
                TransactionId = txId,
                CorrelationId = corrId,
                Description = description,
                IsAnomaly = isAnomaly,
                Scenario = "TechnicalBackground",
                LatencyMs = latencyMs,
                Attributes = new Dictionary<string, object>
                {
                    ["SpikeWindow"] = inSpike,
                    ["FailureWindow"] = inFailure,
                    ["TrendWindow"] = inTrend,
                    ["CpuUsage"] = Math.Round(cpuUsage, 2),
                    ["MemoryUsageMb"] = memUsage,
                    ["RequestSizeBytes"] = reqSize,
                    ["ResponseSizeBytes"] = respSize,
                    ["Retries"] = retries,
                    ["DiskQueueLength"] = diskQueue,
                    ["NetworkErrors"] = netErrors,
                    ["LocalQps"] = Math.Round(localQps, 2),
                    ["IsBackgroundMetric"] = true,
                    ["IsSpikeByLatency"] = isSpikeByLatency,
                    ["IsSpikeByCpu"] = isSpikeByCpu,
                    ["IsSpikeByQps"] = isSpikeByQps
                }
            };

            double jitterFactor = 0.1 + rnd.NextDouble() * 1.5; // 0.1–1.6
            currentTime = currentTime.AddSeconds(
                cfg.AverageInterEventTimeSeconds * jitterFactor);

            counters.TotalEvents++;
            return entry;
        }
    }
}