@0xa5f2e3c4b1d89e76;

struct MemoryNode {
    nodeId @0 :Text;
    nodeLevel @1 :UInt8;  # 0=frame, 1=segment, 2=scene, 3=chapter, 4=meta
    parentId @2 :Text;
    videoId @3 :Text;
    startTime @4 :Float64;
    endTime @5 :Float64;
    summary @6 :Text;
    importance @7 :Float32;
    narrativeTags @8 :List(Text);
    deletedByAI @9 :Text;  # Which AI decided to delete this
    compressionData @10 :CompressionInfo;
    children @11 :List(Text);  # Child node IDs
    
    struct CompressionInfo {
        originalFrames @0 :UInt32;
        keptFrames @1 :UInt32;
        compressionRatio @2 :Float32;
        interpolationPossible @3 :Bool;
        reconstructionQuality @4 :Float32;
    }
}

struct VideoMemory {
    videoId @0 :Text;
    rootNodeId @1 :Text;
    totalDuration @2 :Float64;
    originalFrameCount @3 :UInt32;
    compressedFrameCount @4 :UInt32;
    createdAt @5 :UInt64;
    lastAccessedAt @6 :UInt64;
    metadata @7 :VideoMetadata;
    
    struct VideoMetadata {
        title @0 :Text;
        description @1 :Text;
        fps @2 :Float32;
        resolution @3 :Resolution;
        codec @4 :Text;
        
        struct Resolution {
            width @0 :UInt32;
            height @1 :UInt32;
        }
    }
}