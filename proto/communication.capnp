@0xb7e4f2a3c5d18e92;

struct ProcessRequest {
    videoUrl @0 :Text;
    videoData @1 :Data;  # Alternative to URL
    processingLevel @2 :ProcessingLevel;
    options @3 :ProcessingOptions;
    
    enum ProcessingLevel {
        quick @0;
        standard @1;
        deep @2;
    }
    
    struct ProcessingOptions {
        enablePeripheralDetection @0 :Bool = true;
        compressionLevel @1 :Float32 = 0.8;
        narrativeFocus @2 :List(Text);  # Keywords to focus on
        ignoreStaticScenes @3 :Bool = true;
        audioAnalysis @4 :Bool = true;
    }
}

struct QueryRequest {
    memoryId @0 :Text;
    query @1 :Text;
    detailLevel @2 :UInt8 = 5;  # 1-10
    timeRange @3 :TimeRange;
    
    struct TimeRange {
        startTime @0 :Float64;
        endTime @1 :Float64;
    }
}

struct QueryResponse {
    memoryId @0 :Text;
    results @1 :List(QueryResult);
    
    struct QueryResult {
        nodeId @0 :Text;
        relevanceScore @1 :Float32;
        timestamp @2 :Float64;
        summary @3 :Text;
        context @4 :Text;
        reconstructable @5 :Bool;
    }
}