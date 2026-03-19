//+------------------------------------------------------------------+
//|                                           MA_impulse_logged.mq5 |
//|                                  Copyright 2026, Antigravity     |
//+------------------------------------------------------------------+
#property copyright   "Copyright 2026, Antigravity"
#property link        "https://google.com"
#property version     "1.03"
#property strict

//--- input parameters
input ENUM_MA_METHOD      InpMAMethod          = MODE_SMA;
input int                 InpMAPeriod          = 50;
input ENUM_APPLIED_PRICE  InpPrice             = PRICE_CLOSE;
input int                 InpMAShift           = 0;
input ENUM_TIMEFRAMES     InpTimeframe         = PERIOD_CURRENT;

// tracking parameters (IN PRICE UNITS, e.g. 10.0 = $10 move)
input double              InpThresholdPoints   = 10.0;   
input double              InpReversalPercent   = 30.0; 

input string              InpFileName          = "ma_impulse_data.csv";

// --- Session Inputs (UTC HOURS)
input int                 InpSydneyStart       = 22;
input int                 InpSydneyEnd         = 7;
input int                 InpTokyoStart        = 0;
input int                 InpTokyoEnd          = 9;
input int                 InpLondonStart       = 8;
input int                 InpLondonEnd         = 17;
input int                 InpNYStart           = 13;
input int                 InpNYEnd             = 22;

//--- global variables
int      handleMA;
bool     isFirstBar = true;
datetime lastBarTime = 0;

//--- Wave State Structure
struct WaveState {
   bool     isActive;
   int      type;             // 1 = Bullish, -1 = Bearish
   datetime startTime;
   string   startSession;     
   double   startPrice;
   
   // --- Tracking for Reversal & Impulse Peaks
   bool     isReversed;       // True if 30% reversal was hit
   double   impulsePeakPrice; // The peak that was "Locked" by 30% reversal
   datetime impulsePeakTime;
   double   maAtImpulsePeak;  // Snapshot of MA at impulse peak
   double   reversalPrice;    // Price where move reversed 30% from peak
   datetime reversalTime;
   double   maAtReversal;     // Snapshot of MA at reversal point
   
   // --- Deepest Retracement tracking (The "Low" until peak broken)
   double   absLowPrice;      // The deepest point reached against the trend
   datetime absLowTime;
   double   maAtAbsLow;
   bool     reversalCycleComplete; // Once trough is locked and new high found, stop reversal hunting
   
   double   extremePrice;     // The Absolute Peak for the entire wave duration
   datetime extremeTime;
   double   maPriceAtExtreme; // Snapshot of MA at absolute extreme
} currentWave;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
   handleMA = iMA(_Symbol, InpTimeframe, InpMAPeriod, InpMAShift, InpMAMethod, InpPrice);
   if(handleMA == INVALID_HANDLE) return(INIT_FAILED);
   
   currentWave.isActive = false;
   
   // Update Header with Full Descriptive Names for all 26 columns
   if(!FileIsExist(InpFileName, FILE_COMMON)) {
      int fileHandle = FileOpen(InpFileName, FILE_WRITE|FILE_CSV|FILE_ANSI|FILE_COMMON, ',');
      if(fileHandle != INVALID_HANDLE) {
         FileWrite(fileHandle, 
            "Symbol", "WaveDirection", "MovingAverageMethod", "MovingAveragePeriod", "ChartTimeframe", 
            "CrossoverStartSession", "CrossoverEndSession",
            "CrossoverStartTime", "CrossoverEndTime", 
            "CrossoverStartPrice", "CrossoverEndPrice",
            "AbsolutePeakPrice", "AbsolutePeakTime", "MA_At_AbsolutePeak",
            "ImpulsePeakPrice", "ImpulsePeakTime", "MA_At_ImpulsePeak",
            "ReversalPrice", "ReversalTime", "MA_At_Reversal",
            "DeepestRetracePrice", "DeepestRetraceTime", "MA_At_DeepestRetrace",
            "ReversalTriggered", "DifferencePoints", "DifferencePercent"
         );
         FileClose(fileHandle);
      }
   }

   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason) { IndicatorRelease(handleMA); }

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
   if(currentWave.isActive) UpdateExtreme();

   datetime curBarTime = iTime(_Symbol, InpTimeframe, 0);
   if(curBarTime != lastBarTime) {
      if(isFirstBar) { lastBarTime = curBarTime; isFirstBar = false; return; }

      double maVal[2], closeVal[2], prevMA[1], prevClose[1];
      if(CopyBuffer(handleMA, 0, 1, 1, maVal) < 1 || CopyClose(_Symbol, InpTimeframe, 1, 1, closeVal) < 1) return;
      if(CopyBuffer(handleMA, 0, 2, 1, prevMA) < 1 || CopyClose(_Symbol, InpTimeframe, 2, 1, prevClose) < 1) return;
      
      bool isBullCross = (prevClose[0] <= prevMA[0] && closeVal[0] > maVal[0]);
      bool isBearCross = (prevClose[0] >= prevMA[0] && closeVal[0] < maVal[0]);
      
      if(isBullCross || isBearCross) {
         if(currentWave.isActive) LogWave(iTime(_Symbol, InpTimeframe, 1), closeVal[0]);
         
         currentWave.isActive = true;
         currentWave.type = isBullCross ? 1 : -1;
         currentWave.startTime = iTime(_Symbol, InpTimeframe, 1);
         currentWave.startSession = GetCurrentSessions(currentWave.startTime);
         currentWave.startPrice = closeVal[0];
         
         currentWave.extremePrice = isBullCross ? iHigh(_Symbol, InpTimeframe, 1) : iLow(_Symbol, InpTimeframe, 1);
         currentWave.extremeTime = currentWave.startTime;
         currentWave.maPriceAtExtreme = maVal[0];
         
         // Initialize Impulse tracking
         currentWave.isReversed = false;
         currentWave.impulsePeakPrice = currentWave.extremePrice;
         currentWave.impulsePeakTime = currentWave.extremeTime;
         currentWave.maAtImpulsePeak = currentWave.maPriceAtExtreme;
         currentWave.reversalPrice = 0;
         currentWave.reversalTime = 0;
         currentWave.maAtReversal = 0;
         
         currentWave.absLowPrice = 0;
         currentWave.absLowTime = 0;
         currentWave.maAtAbsLow = 0;
         currentWave.reversalCycleComplete = false;
      }
      lastBarTime = curBarTime;
   }
}

//+------------------------------------------------------------------+
//| Update extreme price during active wave                          |
//+------------------------------------------------------------------+
void UpdateExtreme()
{
   double high = iHigh(_Symbol, InpTimeframe, 0);
   double low  = iLow(_Symbol, InpTimeframe, 0);
   double currentBid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double currentAsk = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double ma[1];
   
   // 1. Always track Absolute Peak (no matter what)
   if(currentWave.type == 1 && currentBid > currentWave.extremePrice) {
      currentWave.extremePrice = currentBid;
      currentWave.extremeTime = TimeCurrent();
      if(CopyBuffer(handleMA, 0, 0, 1, ma) > 0) currentWave.maPriceAtExtreme = ma[0];
   } else if(currentWave.type == -1 && (currentAsk < currentWave.extremePrice || currentWave.extremePrice == 0)) {
      currentWave.extremePrice = currentAsk;
      currentWave.extremeTime = TimeCurrent();
      if(CopyBuffer(handleMA, 0, 0, 1, ma) > 0) currentWave.maPriceAtExtreme = ma[0];
   }

   // 2. Impulse & Reversal Tracking Logic
   if(currentWave.type == 1) { // BULLISH

      if(!currentWave.isReversed && !currentWave.reversalCycleComplete) {
         // === PHASE A: Track the High + Hunt for 30% Reversal ===
         if(currentBid > currentWave.impulsePeakPrice) {
            currentWave.impulsePeakPrice = currentBid;
            currentWave.impulsePeakTime  = TimeCurrent();
            if(CopyBuffer(handleMA, 0, 0, 1, ma) > 0) currentWave.maAtImpulsePeak = ma[0];
         }
         double move = currentWave.impulsePeakPrice - currentWave.startPrice;
         if(move >= InpThresholdPoints) {
            if(currentBid <= currentWave.impulsePeakPrice - (move * InpReversalPercent / 100.0)) {
               // 30% Reversal Triggered - LOCK impulsePeakPrice from here
               currentWave.isReversed    = true;
               currentWave.reversalPrice = currentBid;
               currentWave.reversalTime  = TimeCurrent();
               if(CopyBuffer(handleMA, 0, 0, 1, ma) > 0) currentWave.maAtReversal = ma[0];
               currentWave.absLowPrice   = currentBid;
               currentWave.absLowTime    = TimeCurrent();
               if(CopyBuffer(handleMA, 0, 0, 1, ma) > 0) currentWave.maAtAbsLow = ma[0];
            }
         }

      } else if(currentWave.isReversed) {
         // === PHASE B: Track Trough ONLY. impulsePeakPrice is LOCKED. ===
         if(currentBid < currentWave.absLowPrice) {
            currentWave.absLowPrice = currentBid;
            currentWave.absLowTime  = TimeCurrent();
            if(CopyBuffer(handleMA, 0, 0, 1, ma) > 0) currentWave.maAtAbsLow = ma[0];
         }
         // Breakout: old peak broken -> Trough is locked, switch to Phase C
         if(currentBid > currentWave.impulsePeakPrice) {
            currentWave.isReversed           = false;
            currentWave.reversalCycleComplete = true;
         }

      }
      // === PHASE C: reversalCycleComplete = true ===
      // Do nothing here - extremePrice (AbsolutePeak) already tracks all-time high above.

   } else { // BEARISH

      if(!currentWave.isReversed && !currentWave.reversalCycleComplete) {
         // === PHASE A: Track the Low + Hunt for 30% Reversal ===
         if(currentAsk < currentWave.impulsePeakPrice || currentWave.impulsePeakPrice == 0) {
            currentWave.impulsePeakPrice = currentAsk;
            currentWave.impulsePeakTime  = TimeCurrent();
            if(CopyBuffer(handleMA, 0, 0, 1, ma) > 0) currentWave.maAtImpulsePeak = ma[0];
         }
         double move = currentWave.startPrice - currentWave.impulsePeakPrice;
         if(move >= InpThresholdPoints) {
            if(currentAsk >= currentWave.impulsePeakPrice + (move * InpReversalPercent / 100.0)) {
               // 30% Reversal Triggered - LOCK impulsePeakPrice from here
               currentWave.isReversed    = true;
               currentWave.reversalPrice = currentAsk;
               currentWave.reversalTime  = TimeCurrent();
               if(CopyBuffer(handleMA, 0, 0, 1, ma) > 0) currentWave.maAtReversal = ma[0];
               currentWave.absLowPrice   = currentAsk;
               currentWave.absLowTime    = TimeCurrent();
               if(CopyBuffer(handleMA, 0, 0, 1, ma) > 0) currentWave.maAtAbsLow = ma[0];
            }
         }

      } else if(currentWave.isReversed) {
         // === PHASE B: Track Trough ONLY. impulsePeakPrice is LOCKED. ===
         if(currentAsk > currentWave.absLowPrice) {
            currentWave.absLowPrice = currentAsk;
            currentWave.absLowTime  = TimeCurrent();
            if(CopyBuffer(handleMA, 0, 0, 1, ma) > 0) currentWave.maAtAbsLow = ma[0];
         }
         // Breakout: old peak broken -> Trough is locked, switch to Phase C
         if(currentAsk < currentWave.impulsePeakPrice && currentWave.impulsePeakPrice != 0) {
            currentWave.isReversed           = false;
            currentWave.reversalCycleComplete = true;
         }

      }
      // === PHASE C: reversalCycleComplete = true ===
      // Do nothing here - extremePrice (AbsolutePeak) already tracks all-time low above.
   }
}

//+------------------------------------------------------------------+
//| Log to CSV                                                       |
//+------------------------------------------------------------------+
void LogWave(datetime endTime, double endPrice)
{
   double diff = MathAbs(currentWave.extremePrice - currentWave.startPrice);
   
   double pct = (currentWave.startPrice != 0) ? (diff / currentWave.startPrice) * 100.0 : 0;
   string endSess = GetCurrentSessions(endTime);
   ENUM_TIMEFRAMES tf = (InpTimeframe == PERIOD_CURRENT) ? (ENUM_TIMEFRAMES)Period() : InpTimeframe;
   
   int fileHandle = FileOpen(InpFileName, FILE_READ|FILE_WRITE|FILE_CSV|FILE_ANSI|FILE_COMMON, ',');
   if(fileHandle != INVALID_HANDLE) {
      FileSeek(fileHandle, 0, SEEK_END);
      FileWrite(fileHandle,
         _Symbol, (currentWave.type == 1 ? "BULL" : "BEAR"), EnumToString(InpMAMethod), InpMAPeriod, EnumToString(tf), 
         currentWave.startSession, endSess,
         TimeToString(currentWave.startTime), TimeToString(endTime), 
         NormalizeDouble(currentWave.startPrice, _Digits), NormalizeDouble(endPrice, _Digits),
         NormalizeDouble(currentWave.extremePrice, _Digits), TimeToString(currentWave.extremeTime), NormalizeDouble(currentWave.maPriceAtExtreme, _Digits),
         NormalizeDouble(currentWave.impulsePeakPrice, _Digits), TimeToString(currentWave.impulsePeakTime), NormalizeDouble(currentWave.maAtImpulsePeak, _Digits),
         NormalizeDouble(currentWave.reversalPrice, _Digits), TimeToString(currentWave.reversalTime), NormalizeDouble(currentWave.maAtReversal, _Digits),
         NormalizeDouble(currentWave.absLowPrice, _Digits), TimeToString(currentWave.absLowTime), NormalizeDouble(currentWave.maAtAbsLow, _Digits),
         (currentWave.absLowPrice != 0 ? "YES" : "NO"),
         NormalizeDouble(diff, _Digits), NormalizeDouble(pct, 2)
      );
      FileClose(fileHandle);
      Print("Logged Wave: ", _Symbol, " Type: ", (currentWave.type == 1 ? "Bull" : "Bear"), " | Gap: ", diff);
   } else {
      Print("Failed to open file for logging: ", InpFileName);
   }
   currentWave.isActive = false;
}

//+------------------------------------------------------------------+
//| Session Logic                                                    |
//+------------------------------------------------------------------+
string GetCurrentSessions(datetime timeInput)
{
   datetime local = TimeCurrent();
   datetime gmt = TimeGMT();
   int offset = (int)(local - gmt);
   datetime utc = timeInput - offset;
   MqlDateTime dt; TimeToStruct(utc, dt);
   
   string s = "";
   if(IsTimeInRange(dt.hour, InpSydneyStart, InpSydneyEnd)) s += (s==""?"":"-") + "Sydney";
   if(IsTimeInRange(dt.hour, InpTokyoStart, InpTokyoEnd))   s += (s==""?"":"-") + "Tokyo";
   if(IsTimeInRange(dt.hour, InpLondonStart, InpLondonEnd)) s += (s==""?"":"-") + "London";
   if(IsTimeInRange(dt.hour, InpNYStart, InpNYEnd))         s += (s==""?"":"-") + "NewYork";
   
   if(s == "") return "Closed";
   return s;
}

bool IsTimeInRange(int h, int s, int e) {
   if(s < e) return (h >= s && h < e);
   return (h >= s || h < e);
}
