@0x85150b117366d14b;

struct FrameData {
    frameNumber @0 :UInt32;
    timestamp @1 :Float64;
    data @2 :Data;  # Raw frame bytes
}

struct GapperReport {
    gapperId @0 :Text;
    gapperType @1 :GapperType;
    videoId @2 :Text;
    startFrame @3 :UInt32;
    endFrame @4 :UInt32;
    summary @5 :Text;
    importance @6 :Float32;
    features @7 :Features;
    peripheralAnomalies @8 :List(Anomaly);
    
    enum GapperType {
        frame @0;
        segment @1;
        scene @2;
        chapter @3;
        meta @4;
    }
    
    struct Features {
        motionScore @0 :Float32;
        objects @1 :List(Text);
        sceneType @2 :Text;
        hasAnomaly @3 :Bool;
        audioImportance @4 :Float32;
    }
    
    struct Anomaly {
        boxCoordinates @0 :List(Int32);
        anomalyType @1 :Text;
        confidence @2 :Float32;
    }
}

struct CompressionMap {
    videoId @0 :Text;
    segments @1 :List(Segment);
    
    struct Segment {
        startFrame @0 :UInt32;
        endFrame @1 :UInt32;
        keepFrames @2 :List(UInt32);
        compressionRatio @3 :Float32;
        canInterpolate @4 :Bool;
    }
}