//+------------------------------------------------------------------+
//|                 MA Crossover Impulse Data Collector              |
//|                 Logs impulse in points and percent               |
//|                 This EA does NOT trade                           |
//+------------------------------------------------------------------+
#property strict
#property copyright "You"
#property version   "1.00"

//--- inputs
input ENUM_MA_METHOD      InpMAMethod          = MODE_SMA;
input int                 InpMAPeriod          = 50;
input ENUM_APPLIED_PRICE  InpPrice             = PRICE_CLOSE;
input int                 InpMAShift           = 0;
input ENUM_TIMEFRAMES     InpTimeframe         = PERIOD_CURRENT;

// segment size in discrete price units (e.g. 5.0 = 5 dollars/points of price)
input double              InpSegmentSizePrice  = 5.0;
input string              InpFileName          = "ma_impulse_data.csv";

//--- global variables
int      maHandle        = INVALID_HANDLE;
int      fileHandle      = INVALID_HANDLE;

bool     active          = false;      // tracking after a crossover?
int      crossDirection  = 0;          // +1 bullish, -1 bearish
double   crossPrice      = 0.0;
datetime crossTime       = 0;

double   maxFavorable    = 0.0;        // max favorable move in points
double   maxAdverse      = 0.0;        // max adverse move in points (reserved for future use)
int      lastSegmentLogged = 0;
double   segmentSizePoints = 0.0;      // segment size expressed in points

// per-segment tracking (stored in memory, flushed to CSV when sequence ends)
datetime segmentTimes[];                // time when each segment level was first reached
double   segmentPrices[];               // price when each segment level was first reached

// overall extreme (highest high for bullish, lowest low for bearish)
double   sequenceExtremePrice = 0.0;
datetime sequenceExtremeTime  = 0;

// crossover end time (time when opposite side close happens)
datetime crossEndTime         = 0;

// cached offset between server time and UTC (GMT)
int      serverToUtcOffsetSec = 0;

//+------------------------------------------------------------------+
int OnInit()
{
   //--- cache server->UTC offset (in seconds)
   serverToUtcOffsetSec = (int)(TimeGMT() - TimeCurrent());

   //--- create MA handle
   maHandle = iMA(_Symbol, InpTimeframe,
                  InpMAPeriod, InpMAShift, InpMAMethod, InpPrice);
   if(maHandle == INVALID_HANDLE)
   {
      Print("MA_Impulse_Logger: failed to create MA handle. Error: ", GetLastError());
      return(INIT_FAILED);
   }

   //--- open / create CSV (comma separated) file
   // FILE_READ | FILE_WRITE lets us append without losing existing data.
   fileHandle = FileOpen(InpFileName,
                         FILE_READ | FILE_WRITE | FILE_CSV | FILE_ANSI,
                         ',');
   if(fileHandle == INVALID_HANDLE)
   {
      Print("MA_Impulse_Logger: failed to open file: ", InpFileName,
            " Error: ", GetLastError());
      return(INIT_FAILED);
   }

   //--- if file is new, write header
   if(FileSize(fileHandle) == 0)
   {
      FileWrite(fileHandle,
         "symbol",
         "session",
         "cross_time",
         "cross_end_time",
         "cross_price",
         "cross_type",
         "segment_index",
         "segment_time",
         "segment_price",
         "segment_size_price",
         "segment_move_points",
         "segment_move_percent",
         "segment_direction",
         "sequence_extreme_price",
         "sequence_extreme_time",
         "is_final"
      );
   }

   //--- move to end of file for appending
   FileSeek(fileHandle, 0, SEEK_END);

   Print("MA_Impulse_Logger initialized. Logging to file: ", InpFileName);
   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   if(fileHandle != INVALID_HANDLE)
      FileClose(fileHandle);
   if(maHandle != INVALID_HANDLE)
      IndicatorRelease(maHandle);
}

//+------------------------------------------------------------------+
void OnTick()
{
   static datetime lastBarTime = 0;

   //--- get last 3 bars on selected timeframe
   MqlRates rates[3];
   if(CopyRates(_Symbol, InpTimeframe, 0, 3, rates) != 3)
      return;

   //--- check new bar
   if(rates[0].time != lastBarTime)
   {
      // new bar formed, check crossovers on the closed bar
      CheckForCrossover(rates);
      lastBarTime = rates[0].time;
   }

   //--- if we are tracking an active crossover, update impulse
   if(active)
      UpdateImpulse();
}

//+------------------------------------------------------------------+
//| Detect MA crossovers on closed bars                              |
//+------------------------------------------------------------------+
void CheckForCrossover(const MqlRates &rates[])
{
   double ma[3];
   if(CopyBuffer(maHandle, 0, 0, 3, ma) != 3)
      return;

   // indexes:
   // rates[0] -> current forming bar
   // rates[1] -> just closed bar
   // rates[2] -> previous closed bar
   double close0 = rates[1].close; // just closed
   double ma0    = ma[1];

   // side of price relative to MA for the just‑closed candle
   int side = 0;
   if(close0 > ma0)  side = +1;   // above MA -> bullish side
   if(close0 < ma0)  side = -1;   // below MA -> bearish side

   // if we are not yet tracking anything and we have a clear side, start it
   if(!active && side != 0)
   {
      StartNewCrossover(side, close0, rates[1].time);
      return;
   }

   // if we are tracking and the closed candle is on the opposite side,
   // then the current crossover ENDS here and a new one immediately BEGINS
   if(active && side != 0 && side != crossDirection)
   {
      // record crossover end time at this closed bar
      crossEndTime = rates[1].time;
      // end previous sequence and write its data
      FinalizeSequence(true);
      // start new sequence from this closed candle
      StartNewCrossover(side, close0, rates[1].time);
   }
}

//+------------------------------------------------------------------+
//| Start tracking a new crossover                                   |
//+------------------------------------------------------------------+
void StartNewCrossover(int direction, double price, datetime timeCross)
{
   // if already active, finalize the previous sequence
   if(active)
      FinalizeSequence(false);

   active            = true;
   crossDirection    = direction;
   crossPrice        = price;
   crossTime         = timeCross;
   maxFavorable      = 0.0;
   maxAdverse        = 0.0;
   lastSegmentLogged = 0;

   // compute segment size in points for this sequence
   if(_Point > 0.0 && InpSegmentSizePrice > 0.0)
      segmentSizePoints = InpSegmentSizePrice / _Point;
   else
      segmentSizePoints = 0.0;

   // reset arrays and extremes
   ArrayResize(segmentTimes, 0);
   ArrayResize(segmentPrices, 0);
   sequenceExtremePrice = price;
   sequenceExtremeTime  = timeCross;

   Print("MA_Impulse_Logger: new crossover dir=", direction,
         " price=", DoubleToString(price, _Digits),
         " time=", TimeToString(timeCross, TIME_DATE|TIME_SECONDS));
}

//+------------------------------------------------------------------+
//| Update impulse on every tick                                     |
//+------------------------------------------------------------------+
void UpdateImpulse()
{
   double price = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double point = _Point;

   if(point <= 0.0)
      return;

   // update true extreme using bar high/low of current bar
   double barHigh = iHigh(_Symbol, InpTimeframe, 0);
   double barLow  = iLow(_Symbol,  InpTimeframe, 0);

   // compute favorable move using true extremes (not synthetic levels),
   // and keep track of the max favorable extreme price and its time
   double favorableMovePoints = 0.0;
   if(crossDirection == +1) // bullish
   {
      // update bullish extreme as highest high seen since crossover
      if(barHigh > sequenceExtremePrice || sequenceExtremePrice == 0.0)
      {
         sequenceExtremePrice = barHigh;
         sequenceExtremeTime  = TimeCurrent();
      }

      if(sequenceExtremePrice > crossPrice)
         favorableMovePoints = (sequenceExtremePrice - crossPrice) / point;
   }
   else if(crossDirection == -1) // bearish
   {
      // update bearish extreme as lowest low seen since crossover
      if(barLow < sequenceExtremePrice || sequenceExtremePrice == 0.0)
      {
         sequenceExtremePrice = barLow;
         sequenceExtremeTime  = TimeCurrent();
      }

      if(sequenceExtremePrice < crossPrice)
         favorableMovePoints = (crossPrice - sequenceExtremePrice) / point;
   }

   // store for completeness (not strictly needed for segments now)
   if(favorableMovePoints > maxFavorable) maxFavorable = favorableMovePoints;

   //--- log new favorable segments in memory (segment size is in PRICE units)
   if(segmentSizePoints <= 0.0)
      return;

   int currentSegIndex = (int)(favorableMovePoints / segmentSizePoints);

   while(currentSegIndex > lastSegmentLogged)
   {
      int    segIdx          = lastSegmentLogged + 1;
      double segExtremePoints = (double)segIdx * segmentSizePoints; // distance from crossPrice in points
      // grow arrays and store the true extreme price/time when this segment
      // threshold was first exceeded
      ArrayResize(segmentTimes,  segIdx);
      ArrayResize(segmentPrices, segIdx);
      segmentTimes[segIdx - 1]  = sequenceExtremeTime;
      segmentPrices[segIdx - 1] = sequenceExtremePrice;

      lastSegmentLogged = segIdx;
   }
}

//+------------------------------------------------------------------+
//| Finalize a sequence (e.g. when opposite crossover appears)       |
//| Here called only when starting a new crossover.                  |
//+------------------------------------------------------------------+
void FinalizeSequence(bool oppositeCross)
{
   if(!active)
      return;

   // when the opposite crossover appears, flush all stored segments to CSV.
   // Each segment row contains: cross time/price, segment time/price, and move metrics.
   // If no segment thresholds were ever reached, write a single row for the crossover itself
   // with segment_index = 0. If segments exist, we skip the 0 row and only write segments.
   if(lastSegmentLogged == 0)
   {
      WriteSegment(
         0,                   // segment_index 0 => crossover candle itself
         crossPrice,          // price at crossover (close price)
         crossTime,           // time at crossover (close time)
         0.0,                 // movePoints = 0 at start
         0,                   // FLAT relative direction at start
         1                    // no segments hit, so this row is final
      );
   }
   else
   {
      // Write one row per hit segment in order.
      for(int i = 1; i <= lastSegmentLogged; i++)
      {
         // movement distance in PRICE units (dollars), not broker points
         double movePoints = (double)i * InpSegmentSizePrice;
         // movement is always in favor for these stored segments
         int    dir        = +1;
         int    isFinal    = (i == lastSegmentLogged ? 1 : 0);

         WriteSegment(i, segmentPrices[i - 1], segmentTimes[i - 1], movePoints, dir, isFinal);
      }
   }

   active = false;
}

//+------------------------------------------------------------------+
//| Write one row (segment) to CSV                                   |
//+------------------------------------------------------------------+
void WriteSegment(int segmentIndex,
                  double segmentPrice,
                  datetime segmentTime,
                  double movePoints,
                  int segmentDirection,
                  int isFinal)
{
   if(fileHandle == INVALID_HANDLE)
      return;

   // movePoints is already in price units (dollars)
   double movePrice = movePoints;

   // percent move from crossPrice, using price units directly
   double movePercent = 0.0;
   if(crossPrice != 0.0)
      movePercent = (movePrice / crossPrice) * 100.0;

   // human-readable labels for direction fields
   string crossTypeText;
   if(crossDirection == 1)
      crossTypeText = "BULLISH";
   else if(crossDirection == -1)
      crossTypeText = "BEARISH";
   else
      crossTypeText = "NONE";

   string segmentDirText;
   if(segmentDirection == 1)
      segmentDirText = "FAVOR";
   else if(segmentDirection == -1)
      segmentDirText = "AGAINST";
   else
      segmentDirText = "FLAT";

   // derive session name based on UTC time of the segment
   string sessionName;
   datetime utcTime = segmentTime + serverToUtcOffsetSec;
   MqlDateTime tm;
   TimeToStruct(utcTime, tm);
   int hour = tm.hour;

   // Four main sessions, using standard UTC windows:
   // Sydney: 21:00–06:00, Tokyo: 23:00–08:00, London: 08:00–17:00, New York: 13:00–22:00
   if((hour >= 21 && hour <= 23) || (hour >= 0 && hour < 6))
      sessionName = "SYDNEY";
   else if(hour >= 23 || hour < 8)
      sessionName = "TOKYO";
   else if(hour >= 8 && hour < 13)
      sessionName = "LONDON";
   else if(hour >= 13 && hour < 22)
      sessionName = "NEW_YORK";
   else
      sessionName = "OFF_SESSION";

   FileWrite(fileHandle,
      _Symbol,                                      // symbol
      sessionName,                                  // session
      TimeToString(crossTime, TIME_DATE|TIME_SECONDS), // cross_time
      TimeToString(crossEndTime, TIME_DATE|TIME_SECONDS), // cross_end_time
      DoubleToString(crossPrice, _Digits),           // cross_price
      crossTypeText,                                 // cross_type (BULLISH/BEARISH)
      segmentIndex,                                  // segment_index
      TimeToString(segmentTime, TIME_DATE|TIME_SECONDS), // segment_time
      DoubleToString(segmentPrice, _Digits),         // segment_price
      InpSegmentSizePrice,                           // segment_size_price (price units)
      movePoints,                                    // segment_move_points (PRICE units from cross)
      movePercent,                                   // segment_move_percent
      segmentDirText,                                // segment_direction (FAVOR/AGAINST)
      DoubleToString(sequenceExtremePrice, _Digits), // sequence_extreme_price (max/min before crossover end)
      TimeToString(sequenceExtremeTime, TIME_DATE|TIME_SECONDS), // sequence_extreme_time
      isFinal                                        // is_final
   );

   FileFlush(fileHandle);
}

