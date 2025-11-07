import React, { useState } from 'react';
import { Camera, AlertCircle, CheckCircle, Info, TrendingUp, Shield, Clock, DollarSign, FileText, BarChart3, PieChart, Activity } from 'lucide-react';

const UnderwriteGPTEnhanced = () => {
  const [mode, setMode] = useState('underwriter'); // 'underwriter' or 'customer'
  const [query, setQuery] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [showStory, setShowStory] = useState(false);

  // Simulated AI response (in real app, this would call your backend)
  const analyzeApplication = async (inputQuery) => {
    setLoading(true);
    
    // Simulate API delay
    await new Promise(resolve => setTimeout(resolve, 2000));
    
    // Extract features from query
    const features = extractFeatures(inputQuery);
    
    // Calculate risk using your validated model
    const riskAnalysis = calculateEnhancedRisk(features);
    
    // Find similar cases (simulated)
    const similarCases = generateSimilarCases(features, riskAnalysis);
    
    // Make decision
    const decision = makeSmartDecision(riskAnalysis, similarCases, features);
    
    // Generate LLM-style explanation
    const narrative = generateNarrative(decision, riskAnalysis, similarCases, features);
    
    setResult({
      decision,
      riskAnalysis,
      similarCases,
      features,
      narrative
    });
    
    setLoading(false);
  };

  const extractFeatures = (text) => {
    const lowerText = text.toLowerCase();
    
    // Extract age
    const ageMatch = lowerText.match(/(\d+)[\s-]?(?:year|yo|years old)/);
    const customerAge = ageMatch ? parseInt(ageMatch[1]) : 35;
    
    // Extract vehicle age
    const vAgeMatch = lowerText.match(/(\d+)[\s-]?year[\s-]?old\s+(?:car|vehicle)/);
    const vehicleAge = vAgeMatch ? parseFloat(vAgeMatch[1]) : 5;
    
    // Extract subscription
    const subMatch = lowerText.match(/(\d+)[\s-]?month/);
    const subscription = subMatch ? parseInt(subMatch[1]) : 6;
    
    // Extract airbags
    const airbagMatch = lowerText.match(/(\d+)\s*airbag/);
    const airbags = airbagMatch ? parseInt(airbagMatch[1]) : 4;
    
    // Safety features
    const hasESC = lowerText.includes('esc') && !lowerText.includes('no esc');
    const hasBrakeAssist = lowerText.includes('brake assist');
    const hasTPMS = lowerText.includes('tpms');
    
    // Region
    const isUrban = lowerText.includes('urban') || lowerText.includes('city');
    
    // Fuel type
    let fuelType = 'Petrol';
    if (lowerText.includes('diesel')) fuelType = 'Diesel';
    if (lowerText.includes('cng')) fuelType = 'CNG';
    
    return {
      customerAge,
      vehicleAge,
      subscription,
      airbags,
      hasESC,
      hasBrakeAssist,
      hasTPMS,
      isUrban,
      fuelType
    };
  };

  const calculateEnhancedRisk = (features) => {
    // YOUR VALIDATED WEIGHTS from feature engineering
    const weights = {
      subscription: 0.507,  // Highest correlation
      driver: 0.143,
      region: 0.139,
      vehicle: 0.123,
      safety: 0.088
    };
    
    // Subscription risk (most important!)
    let subscriptionRisk = 0.50;
    let subCategory = 'MODERATE';
    if (features.subscription < 3) {
      subscriptionRisk = 0.85;
      subCategory = 'VERY HIGH';
    } else if (features.subscription < 6) {
      subscriptionRisk = 0.65;
      subCategory = 'HIGH';
    } else if (features.subscription >= 9) {
      subscriptionRisk = 0.30;
      subCategory = 'LOW';
    }
    
    // Driver risk
    let driverRisk = 0.50;
    let driverCategory = 'MODERATE';
    if (features.customerAge < 25) {
      driverRisk = 0.75;
      driverCategory = 'HIGH';
    } else if (features.customerAge < 30) {
      driverRisk = 0.60;
      driverCategory = 'MODERATE';
    } else if (features.customerAge > 65) {
      driverRisk = 0.70;
      driverCategory = 'HIGH';
    } else if (features.customerAge >= 35 && features.customerAge <= 55) {
      driverRisk = 0.35;
      driverCategory = 'LOW';
    }
    
    // Vehicle risk
    let vehicleRisk = 0.50;
    let vehicleCategory = 'MODERATE';
    if (features.vehicleAge <= 3) {
      vehicleRisk = 0.35;
      vehicleCategory = 'LOW';
    } else if (features.vehicleAge <= 7) {
      vehicleRisk = 0.50;
      vehicleCategory = 'MODERATE';
    } else {
      vehicleRisk = 0.75;
      vehicleCategory = 'HIGH';
    }
    
    // Region risk
    const regionRisk = features.isUrban ? 0.55 : 0.45;
    const regionCategory = features.isUrban ? 'MODERATE' : 'LOW';
    
    // Safety risk
    const safetyScore = (
      (features.airbags / 6) +
      (features.hasESC ? 1 : 0) +
      (features.hasBrakeAssist ? 1 : 0) +
      (features.hasTPMS ? 1 : 0)
    ) / 4;
    const safetyRisk = 1 - safetyScore;
    const safetyCategory = safetyRisk < 0.4 ? 'LOW' : safetyRisk < 0.6 ? 'MODERATE' : 'HIGH';
    
    // Calculate weighted overall risk
    const overallRisk = 
      weights.subscription * subscriptionRisk +
      weights.driver * driverRisk +
      weights.vehicle * vehicleRisk +
      weights.region * regionRisk +
      weights.safety * safetyRisk;
    
    // Determine overall category based on your data distribution
    let overallCategory = 'MODERATE';
    if (overallRisk < 0.4) overallCategory = 'LOW';
    else if (overallRisk >= 0.4 && overallRisk < 0.6) overallCategory = 'MODERATE';
    else if (overallRisk >= 0.6 && overallRisk < 0.75) overallCategory = 'HIGH';
    else overallCategory = 'VERY HIGH';
    
    return {
      overallRisk,
      overallCategory,
      components: {
        subscription: { risk: subscriptionRisk, weight: weights.subscription, category: subCategory },
        driver: { risk: driverRisk, weight: weights.driver, category: driverCategory },
        vehicle: { risk: vehicleRisk, weight: weights.vehicle, category: vehicleCategory },
        region: { risk: regionRisk, weight: weights.region, category: regionCategory },
        safety: { risk: safetyRisk, weight: weights.safety, category: safetyCategory }
      }
    };
  };

  const generateSimilarCases = (features, riskAnalysis) => {
    // Simulate finding similar cases
    // In reality, you'd query your FAISS index
    const cases = [];
    const baseClaimRate = 0.064; // 6.4%
    
    // Adjust claim probability based on risk
    // Key insight: Even high-risk profiles might have many no-claims
    // But the RATE is higher
    let claimProbability = baseClaimRate * (1 + (riskAnalysis.overallRisk - 0.5) * 3);
    claimProbability = Math.max(0.01, Math.min(0.35, claimProbability));
    
    for (let i = 0; i < 10; i++) {
      const similarity = 0.95 - (i * 0.03) + (Math.random() * 0.02);
      const hasClaim = Math.random() < claimProbability;
      
      cases.push({
        id: `CASE-${1000 + i}`,
        similarity,
        age: features.customerAge + (Math.random() * 6 - 3),
        vehicleAge: features.vehicleAge + (Math.random() * 2 - 1),
        subscription: features.subscription + (Math.random() * 2 - 1),
        hasClaim,
        claimAmount: hasClaim ? Math.round(5000 + Math.random() * 8000) : 0,
        risk: riskAnalysis.overallRisk + (Math.random() * 0.1 - 0.05)
      });
    }
    
    return cases;
  };

  const makeSmartDecision = (riskAnalysis, similarCases, features) => {
    const overallRisk = riskAnalysis.overallRisk;
    const claimedCases = similarCases.filter(c => c.hasClaim).length;
    const claimRate = claimedCases / similarCases.length;
    
    // SMART HANDLING OF IMBALANCE:
    // Even if most similar cases show no claims, we consider:
    // 1. The RATE compared to base rate (6.4%)
    // 2. The feature-based risk score
    // 3. The concentration of claims in top-5 most similar
    
    const top5Claims = similarCases.slice(0, 5).filter(c => c.hasClaim).length;
    const riskMultiplier = overallRisk / 0.064; // vs base rate
    
    // Decision logic that accounts for imbalance
    let tier, action, emoji, color, premium, confidence, reasoning;
    
    if (overallRisk < 0.40 && claimRate < 0.10 && top5Claims === 0) {
      tier = 'APPROVE';
      action = 'Standard Approval';
      emoji = '✅';
      color = '#10b981';
      premium = 'Standard rates - no loading';
      confidence = 92;
      reasoning = `Excellent risk profile! Your risk score (${(overallRisk*100).toFixed(1)}%) is well below our threshold. Among 10 similar policies, only ${claimedCases} resulted in claims. This is ${riskMultiplier.toFixed(1)}x the baseline, indicating you're a lower-risk driver.`;
    } else if (overallRisk < 0.50 && claimRate < 0.15) {
      tier = 'MONITOR';
      action = 'Approve with Monitoring';
      emoji = '⚠️';
      color = '#f59e0b';
      premium = '+15-20% premium loading';
      confidence = 78;
      reasoning = `Good risk profile with minor concerns. Your risk score (${(overallRisk*100).toFixed(1)}%) is moderate. Among similar cases, ${claimedCases} out of 10 filed claims (${(claimRate*100).toFixed(0)}%). While most similar drivers didn't claim, the rate is ${(claimRate/0.064).toFixed(1)}x our baseline, suggesting careful monitoring.`;
    } else if (overallRisk < 0.65 && claimRate < 0.30) {
      tier = 'CONDITIONAL';
      action = 'Conditional Approval';
      emoji = '🔶';
      color = '#f97316';
      premium = '+30-45% premium loading';
      confidence = 65;
      reasoning = `Elevated risk detected. Your risk score (${(overallRisk*100).toFixed(1)}%) exceeds our comfort zone. Among similar policies, ${claimedCases} filed claims. Here's the key insight: while ${10-claimedCases} didn't claim, the ${(claimRate*100).toFixed(0)}% rate is ${(claimRate/0.064).toFixed(1)}x higher than average drivers, requiring conditions to offset potential losses.`;
    } else {
      tier = 'REJECT';
      action = 'Decline Application';
      emoji = '❌';
      color = '#ef4444';
      premium = 'Not applicable';
      confidence = 85;
      reasoning = `High risk profile unsuitable for standard coverage. Your risk score (${(overallRisk*100).toFixed(1)}%) significantly exceeds acceptable thresholds. ${claimedCases} out of 10 similar policies resulted in claims (${(claimRate*100).toFixed(0)}%). This ${(claimRate/0.064).toFixed(1)}x multiplier over baseline indicates unacceptable exposure.`;
    }
    
    // Add specific guidance for improvement
    const improvements = [];
    if (riskAnalysis.components.subscription.category !== 'LOW') {
      improvements.push(`Extend your subscription to 12+ months (could reduce premium by 15-20%)`);
    }
    if (riskAnalysis.components.safety.category !== 'LOW') {
      improvements.push(`Add more safety features like ESC or additional airbags (5-10% reduction)`);
    }
    if (features.vehicleAge > 7) {
      improvements.push(`Consider a newer vehicle (3-7 years old) for better rates`);
    }
    
    return {
      tier,
      action,
      emoji,
      color,
      premium,
      confidence,
      reasoning,
      improvements,
      claimedCases,
      claimRate,
      riskMultiplier
    };
  };

  const generateNarrative = (decision, riskAnalysis, similarCases, features) => {
    // LLM-style conversational narrative
    const stories = {
      APPROVE: [
        `Great news! I've analyzed your application and compared it with 10 similar drivers in our database. Here's what I found:`,
        `Your profile shows ${decision.claimedCases} claims out of 10 similar cases - that's only ${(decision.claimRate*100).toFixed(0)}%! This is actually ${decision.riskMultiplier < 1 ? 'below' : 'near'} our baseline of 6.4%, which is excellent.`,
        `The strongest factor in your favor is your ${riskAnalysis.components.subscription.category.toLowerCase()} subscription risk (${features.subscription} months). This shows commitment, and our data proves committed customers are more careful drivers.`,
        `Your age group (${features.customerAge} years) falls in the sweet spot where experience meets caution. Combined with your vehicle's safety features, you're set up for success.`,
        `Bottom line: You're approved at standard rates. Welcome aboard! 🎉`
      ],
      MONITOR: [
        `I've reviewed your application carefully. Here's my assessment:`,
        `Looking at 10 drivers with similar profiles, ${decision.claimedCases} filed claims. That's ${(decision.claimRate*100).toFixed(0)}% - higher than our 6.4% baseline, but not dramatically so.`,
        `Here's what's influencing your rate: Your ${riskAnalysis.components.subscription.category.toLowerCase()} subscription length (${features.subscription} months) is the biggest factor. Shorter subscriptions correlate with ${((riskAnalysis.components.subscription.risk - 0.5) * 200).toFixed(0)}% higher claims in our data.`,
        `The good news? ${similarCases.filter(c => !c.hasClaim).length} similar drivers had no issues at all. We're confident you'll join them with proper monitoring.`,
        `I'm recommending approval with a ${decision.premium}. Think of it as an investment in building your track record with us!`
      ],
      CONDITIONAL: [
        `Thank you for your application. I need to be transparent about what I'm seeing in the data:`,
        `Among 10 drivers most similar to you, ${decision.claimedCases} filed claims - that's ${(decision.claimRate*100).toFixed(0)}%. While ${10-decision.claimedCases} had clean records, the claim rate is ${decision.riskMultiplier.toFixed(1)}x our baseline.`,
        `The primary concern is your ${features.subscription}-month subscription. Our analysis of 58,000+ policies shows this is the #1 predictor of claims, carrying ${(riskAnalysis.components.subscription.weight*100).toFixed(0)}% weight in our model.`,
        `${features.customerAge < 30 ? `Your age group also shows elevated risk patterns, but this improves significantly as you gain more driving experience.` : features.vehicleAge > 7 ? `Your vehicle's age (${features.vehicleAge} years) adds additional risk due to potential mechanical issues.` : ''}`,
        `Here's what I can offer: Conditional approval with ${decision.premium}. Not ideal, but here's how to improve it...`
      ],
      REJECT: [
        `I appreciate your application, and I want to be straightforward about my assessment:`,
        `After analyzing 10 similar profiles, ${decision.claimedCases} resulted in claims - that's ${(decision.claimRate*100).toFixed(0)}%. This is ${decision.riskMultiplier.toFixed(1)}x higher than what we see with average drivers (6.4%).`,
        `The data tells a clear story: Your current profile shows multiple high-risk indicators that, when combined, create unacceptable exposure for standard coverage.`,
        `The biggest concern is ${riskAnalysis.overallCategory === 'VERY HIGH' ? 'your overall risk score' : 'the combination of factors'} - particularly the ${features.subscription}-month subscription and ${features.customerAge < 25 ? 'limited driving experience' : features.vehicleAge > 10 ? 'older vehicle age' : 'urban location'}.`,
        `However, this isn't permanent! Let me show you exactly what changes would make you approvable...`
      ]
    };
    
    return stories[decision.tier];
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (query.trim()) {
      analyzeApplication(query);
      setShowStory(true);
    }
  };

  const quickScenarios = {
    underwriter: [
      { label: 'Low Risk', icon: CheckCircle, query: '42-year-old driver, 3-year-old Petrol sedan, 6 airbags, ESC, brake assist, TPMS, rural area, 12 month subscription', color: '#10b981' },
      { label: 'Medium Risk', icon: AlertCircle, query: '35-year-old driver, 6-year-old Diesel vehicle, 4 airbags, ESC, urban area, 6 month subscription', color: '#f59e0b' },
      { label: 'High Risk', icon: AlertCircle, query: '23-year-old driver, 11-year-old Petrol car, 2 airbags, no ESC, urban area, 3 month subscription', color: '#ef4444' }
    ],
    customer: [
      { label: 'My Sedan', icon: Camera, query: '38 years old, 4-year-old automatic sedan, 4 airbags, ESC, brake assist, city area, 12 month policy', color: '#3b82f6' },
      { label: 'My SUV', icon: Shield, query: '45 years old, 2-year-old diesel SUV, 6 airbags, ESC, brake assist, TPMS, suburban area, 12 month policy', color: '#8b5cf6' },
      { label: 'My Hatchback', icon: Activity, query: '28 years old, 5-year-old petrol hatchback, 2 airbags, no ESC, urban area, 6 month policy', color: '#ec4899' }
    ]
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-50">
      {/* Unique Glassmorphism Header */}
      <div className="relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-r from-blue-600 via-indigo-600 to-purple-600 opacity-90"></div>
        <div className="absolute inset-0" style={{
          backgroundImage: 'radial-gradient(circle at 20% 50%, rgba(255,255,255,0.1) 0%, transparent 50%), radial-gradient(circle at 80% 80%, rgba(255,255,255,0.15) 0%, transparent 50%)',
        }}></div>
        
        <div className="relative max-w-7xl mx-auto px-6 py-8">
          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center gap-4">
              <div className="w-16 h-16 bg-white/20 backdrop-blur-lg rounded-2xl flex items-center justify-center border border-white/30 shadow-xl">
                <Shield className="w-9 h-9 text-white" />
              </div>
              <div>
                <h1 className="text-4xl font-bold text-white tracking-tight">UnderwriteGPT</h1>
                <p className="text-blue-100 text-sm mt-1">AI-Powered Risk Assessment • 58K+ Policy Intelligence</p>
              </div>
            </div>
            
            {/* Mode Toggle */}
            <div className="flex gap-2 bg-white/10 backdrop-blur-lg p-2 rounded-xl border border-white/20">
              <button
                onClick={() => setMode('underwriter')}
                className={`px-6 py-3 rounded-lg font-semibold transition-all ${
                  mode === 'underwriter' 
                    ? 'bg-white text-indigo-600 shadow-lg' 
                    : 'text-white hover:bg-white/10'
                }`}
              >
                <FileText className="w-4 h-4 inline mr-2" />
                Underwriter Mode
              </button>
              <button
                onClick={() => setMode('customer')}
                className={`px-6 py-3 rounded-lg font-semibold transition-all ${
                  mode === 'customer' 
                    ? 'bg-white text-indigo-600 shadow-lg' 
                    : 'text-white hover:bg-white/10'
                }`}
              >
                <Camera className="w-4 h-4 inline mr-2" />
                My Car
              </button>
            </div>
          </div>
          
          {/* Mode-specific description */}
          <div className="bg-white/10 backdrop-blur-lg rounded-xl p-4 border border-white/20">
            {mode === 'underwriter' ? (
              <p className="text-white/90 text-sm">
                <strong>Professional underwriting analysis:</strong> Analyze applications with full evidence, similar case comparisons, and regulatory-compliant decision trails.
              </p>
            ) : (
              <p className="text-white/90 text-sm">
                <strong>Check your own car insurance eligibility:</strong> See if you'd be approved, understand your premium, and get actionable tips to improve your rate!
              </p>
            )}
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-6 py-8">
        {/* Quick Scenarios */}
        {!result && (
          <div className="mb-8">
            <h3 className="text-lg font-semibold text-gray-800 mb-4">
              {mode === 'underwriter' ? '⚡ Quick Test Scenarios' : '🚗 Try These Examples'}
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {quickScenarios[mode].map((scenario, idx) => {
                const Icon = scenario.icon;
                return (
                  <button
                    key={idx}
                    onClick={() => {
                      setQuery(scenario.query);
                      analyzeApplication(scenario.query);
                      setShowStory(true);
                    }}
                    className="p-4 bg-white rounded-xl shadow-md hover:shadow-xl transition-all border-2 border-transparent hover:border-indigo-300 text-left"
                    style={{ borderLeftColor: scenario.color, borderLeftWidth: '4px' }}
                  >
                    <div className="flex items-center gap-3 mb-2">
                      <Icon className="w-5 h-5" style={{ color: scenario.color }} />
                      <span className="font-semibold text-gray-800">{scenario.label}</span>
                    </div>
                    <p className="text-xs text-gray-600 line-clamp-2">{scenario.query}</p>
                  </button>
                );
              })}
            </div>
          </div>
        )}

        {/* Input Form */}
        <form onSubmit={handleSubmit} className="mb-8">
          <div className="bg-white rounded-2xl shadow-xl p-6 border border-gray-200">
            <label className="block text-sm font-semibold text-gray-700 mb-3">
              {mode === 'underwriter' ? '📋 Describe the insurance application:' : '🚗 Tell me about your car and driving profile:'}
            </label>
            <textarea
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder={mode === 'underwriter' 
                ? "Example: 35-year-old driver, 4-year-old Petrol sedan, 4 airbags, ESC, brake assist, urban area, 8 month subscription"
                : "Example: I'm 32 years old, drive a 3-year-old automatic sedan with 4 airbags and ESC in the city, looking for a 12-month policy"
              }
              className="w-full p-4 border-2 border-gray-200 rounded-xl focus:border-indigo-500 focus:ring-2 focus:ring-indigo-200 transition-all resize-none"
              rows="4"
            />
            <div className="flex gap-3 mt-4">
              <button
                type="submit"
                disabled={loading || !query.trim()}
                className="flex-1 bg-gradient-to-r from-indigo-600 to-purple-600 text-white px-6 py-4 rounded-xl font-semibold hover:from-indigo-700 hover:to-purple-700 transition-all shadow-lg hover:shadow-xl disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {loading ? (
                  <span className="flex items-center justify-center gap-2">
                    <Activity className="w-5 h-5 animate-spin" />
                    Analyzing with AI...
                  </span>
                ) : (
                  <span className="flex items-center justify-center gap-2">
                    <TrendingUp className="w-5 h-5" />
                    {mode === 'underwriter' ? 'Analyze Application' : 'Check My Eligibility'}
                  </span>
                )}
              </button>
              
              {result && (
                <button
                  type="button"
                  onClick={() => {
                    setResult(null);
                    setQuery('');
                    setShowStory(false);
                  }}
                  className="px-6 py-4 bg-gray-100 text-gray-700 rounded-xl font-semibold hover:bg-gray-200 transition-all"
                >
                  New Analysis
                </button>
              )}
            </div>
          </div>
        </form>

        {/* Results Section */}
        {result && showStory && (
          <div className="space-y-6 animate-fadeIn">
            {/* Decision Card - Unique Design */}
            <div 
              className="rounded-2xl shadow-2xl overflow-hidden border-4"
              style={{ borderColor: result.decision.color }}
            >
              <div 
                className="p-6"
                style={{ 
                  background: `linear-gradient(135deg, ${result.decision.color}15 0%, ${result.decision.color}05 100%)`
                }}
              >
                <div className="flex items-start justify-between mb-6">
                  <div className="flex-1">
                    <div className="flex items-center gap-3 mb-3">
                      <span className="text-5xl">{result.decision.emoji}</span>
                      <div>
                        <h2 className="text-3xl font-bold" style={{ color: result.decision.color }}>
                          {result.decision.action}
                        </h2>
                        <p className="text-gray-600 mt-1">{result.decision.premium}</p>
                      </div>
                    </div>
                  </div>
                  
                  <div className="text-right">
                    <div className="text-5xl font-bold" style={{ color: result.decision.color }}>
                      {result.decision.confidence}%
                    </div>
                    <div className="text-sm text-gray-600 font-semibold">Confidence</div>
                  </div>
                </div>
                
                {/* Narrative Story */}
                <div className="bg-white rounded-xl p-6 shadow-inner space-y-4">
                  <div className="flex items-start gap-3">
                    <Info className="w-6 h-6 text-indigo-600 flex-shrink-0 mt-1" />
                    <div className="space-y-3">
                      {result.narrative.map((paragraph, idx) => (
                        <p key={idx} className="text-gray-700 leading-relaxed">
                          {paragraph}
                        </p>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Risk Metrics - Storytelling Cards */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="bg-white rounded-xl p-5 shadow-lg border-l-4 border-indigo-500">
                <div className="text-gray-600 text-sm font-semibold mb-2">Overall Risk Score</div>
                <div className="text-3xl font-bold text-indigo-600">
                  {(result.riskAnalysis.overallRisk * 100).toFixed(1)}%
                </div>
                <div className="text-xs text-gray-500 mt-1">{result.riskAnalysis.overallCategory}</div>
              </div>
              
              <div className="bg-white rounded-xl p-5 shadow-lg border-l-4 border-orange-500">
                <div className="text-gray-600 text-sm font-semibold mb-2">Similar Cases</div>
                <div className="text-3xl font-bold text-orange-600">
                  {result.decision.claimedCases}/10
                </div>
                <div className="text-xs text-gray-500 mt-1">Filed Claims</div>
              </div>
              
              <div className="bg-white rounded-xl p-5 shadow-lg border-l-4 border-purple-500">
                <div className="text-gray-600 text-sm font-semibold mb-2">Claim Rate</div>
                <div className="text-3xl font-bold text-purple-600">
                  {(result.decision.claimRate * 100).toFixed(0)}%
                </div>
                <div className="text-xs text-gray-500 mt-1">vs 6.4% baseline</div>
              </div>
              
              <div className="bg-white rounded-xl p-5 shadow-lg border-l-4 border-pink-500">
                <div className="text-gray-600 text-sm font-semibold mb-2">Risk Multiplier</div>
                <div className="text-3xl font-bold text-pink-600">
                  {result.decision.riskMultiplier.toFixed(1)}x
                </div>
                <div className="text-xs text-gray-500 mt-1">vs average driver</div>
              </div>
            </div>

            {/* Risk Components - Visual Story */}
            <div className="bg-white rounded-2xl shadow-xl p-6 border border-gray-200">
              <h3 className="text-xl font-bold text-gray-800 mb-6 flex items-center gap-2">
                <BarChart3 className="w-6 h-6 text-indigo-600" />
                Risk Breakdown Story - What's Driving Your Score?
              </h3>
              
              <div className="space-y-4">
                {Object.entries(result.riskAnalysis.components).map(([key, data]) => {
                  const contribution = data.risk * data.weight * 100;
                  const colors = {
                    LOW: '#10b981',
                    MODERATE: '#f59e0b',
                    HIGH: '#f97316',
                    'VERY HIGH': '#ef4444'
                  };
                  const color = colors[data.category];
                  
                  return (
                    <div key={key} className="space-y-2">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <span className="text-sm font-semibold text-gray-700 capitalize">
                            {key === 'subscription' && '📅 '}
                            {key === 'driver' && '👤 '}
                            {key === 'vehicle' && '🚗 '}
                            {key === 'region' && '📍 '}
                            {key === 'safety' && '🛡️ '}
                            {key}
                          </span>
                          <span 
                            className="text-xs px-2 py-1 rounded-full font-semibold"
                            style={{ backgroundColor: `${color}20`, color }}
                          >
                            {data.category}
                          </span>
                        </div>
                        <div className="text-right">
                          <span className="text-sm font-bold" style={{ color }}>
                            +{contribution.toFixed(1)}%
                          </span>
                          <span className="text-xs text-gray-500 ml-2">
                            (weight: {(data.weight * 100).toFixed(0)}%)
                          </span>
                        </div>
                      </div>
                      
                      <div className="relative h-3 bg-gray-100 rounded-full overflow-hidden">
                        <div 
                          className="h-full rounded-full transition-all duration-1000"
                          style={{ 
                            width: `${(data.risk * 100)}%`,
                            background: `linear-gradient(90deg, ${color} 0%, ${color}88 100%)`
                          }}
                        />
                      </div>
                      
                      {key === 'subscription' && (
                        <p className="text-xs text-gray-600 italic">
                          ⭐ Most important factor! {result.features.subscription}-month subscription has {(data.weight * 100).toFixed(0)}% weight in final score.
                        </p>
                      )}
                    </div>
                  );
                })}
              </div>
              
              <div className="mt-6 p-4 bg-blue-50 rounded-xl border-l-4 border-blue-500">
                <p className="text-sm text-gray-700">
                  <strong>💡 Reading this chart:</strong> Each bar shows how much that factor contributes to your final risk score. 
                  Subscription length matters most ({(result.riskAnalysis.components.subscription.weight * 100).toFixed(0)}% weight) 
                  because our analysis of 58K+ policies found it's the #1 predictor of claims.
                </p>
              </div>
            </div>

            {/* Similar Cases - Storytelling Visualization */}
            <div className="bg-white rounded-2xl shadow-xl p-6 border border-gray-200">
              <h3 className="text-xl font-bold text-gray-800 mb-4 flex items-center gap-2">
                <PieChart className="w-6 h-6 text-indigo-600" />
                Historical Evidence - The Data Behind the Decision
              </h3>
              
              <div className="mb-6 p-4 bg-gradient-to-r from-indigo-50 to-purple-50 rounded-xl">
                <p className="text-sm text-gray-700 leading-relaxed">
                  <strong>Understanding the imbalance:</strong> Most drivers (93.6%) never file claims - that's expected! 
                  What matters is the <strong>rate</strong> among similar profiles. Your {result.decision.claimedCases} claims 
                  out of 10 ({(result.decision.claimRate * 100).toFixed(0)}%) is {' '}
                  {result.decision.riskMultiplier < 1.5 ? (
                    <span className="text-green-600 font-semibold">better than average ✓</span>
                  ) : result.decision.riskMultiplier < 3 ? (
                    <span className="text-orange-600 font-semibold">above average ⚠</span>
                  ) : (
                    <span className="text-red-600 font-semibold">significantly elevated ⚠</span>
                  )}
                </p>
              </div>
              
              <div className="grid grid-cols-10 gap-2 mb-4">
                {result.similarCases.map((case_, idx) => (
                  <div key={idx} className="text-center">
                    <div 
                      className={`w-full aspect-square rounded-lg flex items-center justify-center text-2xl ${
                        case_.hasClaim 
                          ? 'bg-red-100 border-2 border-red-400' 
                          : 'bg-green-100 border-2 border-green-400'
                      }`}
                      title={`Case ${idx + 1}: ${case_.hasClaim ? 'Claimed  + case_.claimAmount : 'No claim'}`}
                    >
                      {case_.hasClaim ? '❌' : '✅'}
                    </div>
                    <div className="text-xs text-gray-500 mt-1">#{idx + 1}</div>
                  </div>
                ))}
              </div>
              
              <div className="grid grid-cols-2 gap-4 mb-4">
                <div className="p-4 bg-green-50 rounded-xl border-l-4 border-green-500">
                  <div className="text-3xl font-bold text-green-600">
                    {result.similarCases.filter(c => !c.hasClaim).length}
                  </div>
                  <div className="text-sm text-gray-600">Clean Records</div>
                  <div className="text-xs text-gray-500 mt-1">
                    These drivers prove your profile CAN succeed
                  </div>
                </div>
                
                <div className="p-4 bg-red-50 rounded-xl border-l-4 border-red-500">
                  <div className="text-3xl font-bold text-red-600">
                    {result.decision.claimedCases}
                  </div>
                  <div className="text-sm text-gray-600">Filed Claims</div>
                  <div className="text-xs text-gray-500 mt-1">
                    Average claim: ${Math.round(result.similarCases.filter(c => c.hasClaim).reduce((sum, c) => sum + c.claimAmount, 0) / result.decision.claimedCases || 0).toLocaleString()}
                  </div>
                </div>
              </div>
              
              <div className="space-y-2">
                <h4 className="font-semibold text-gray-700 text-sm">Top 5 Most Similar Cases:</h4>
                {result.similarCases.slice(0, 5).map((case_, idx) => (
                  <div 
                    key={idx}
                    className={`p-3 rounded-lg border-l-4 ${
                      case_.hasClaim 
                        ? 'bg-red-50 border-red-400' 
                        : 'bg-green-50 border-green-400'
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <span className="text-lg">{case_.hasClaim ? '❌' : '✅'}</span>
                        <div>
                          <div className="text-sm font-semibold text-gray-800">
                            {case_.id} - {Math.round(case_.age)}yo, {case_.vehicleAge.toFixed(1)}yr vehicle
                          </div>
                          <div className="text-xs text-gray-600">
                            {case_.subscription.toFixed(0)}-month subscription • Risk: {(case_.risk * 100).toFixed(1)}%
                          </div>
                        </div>
                      </div>
                      <div className="text-right">
                        <div className="text-sm font-bold text-indigo-600">
                          {(case_.similarity * 100).toFixed(1)}%
                        </div>
                        <div className="text-xs text-gray-500">match</div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Improvements Section */}
            {result.decision.improvements.length > 0 && (
              <div className="bg-gradient-to-br from-emerald-50 to-teal-50 rounded-2xl shadow-xl p-6 border-2 border-emerald-200">
                <h3 className="text-xl font-bold text-gray-800 mb-4 flex items-center gap-2">
                  <TrendingUp className="w-6 h-6 text-emerald-600" />
                  How to Improve Your Premium - Actionable Steps
                </h3>
                
                <div className="space-y-3">
                  {result.decision.improvements.map((improvement, idx) => (
                    <div key={idx} className="flex items-start gap-3 p-4 bg-white rounded-xl shadow-sm">
                      <div className="w-8 h-8 bg-emerald-100 rounded-full flex items-center justify-center flex-shrink-0">
                        <span className="text-emerald-600 font-bold">{idx + 1}</span>
                      </div>
                      <div className="flex-1">
                        <p className="text-gray-700">{improvement}</p>
                      </div>
                      <CheckCircle className="w-5 h-5 text-emerald-500 flex-shrink-0" />
                    </div>
                  ))}
                </div>
                
                <div className="mt-4 p-4 bg-white rounded-xl">
                  <p className="text-sm text-gray-600">
                    <strong className="text-emerald-600">💰 Potential savings:</strong> Implementing all suggestions could reduce your premium by 25-35%, 
                    potentially saving you ${Math.round(1500 * 0.30)}-${Math.round(1500 * 0.35)} annually on a typical $1,500 policy.
                  </p>
                </div>
              </div>
            )}

            {/* Feature Extraction Details */}
            <div className="bg-white rounded-2xl shadow-xl p-6 border border-gray-200">
              <h3 className="text-xl font-bold text-gray-800 mb-4 flex items-center gap-2">
                <Info className="w-6 h-6 text-indigo-600" />
                Extracted Application Details
              </h3>
              
              <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                <div className="p-4 bg-blue-50 rounded-xl">
                  <div className="text-xs text-gray-600 font-semibold mb-1">Driver Age</div>
                  <div className="text-2xl font-bold text-blue-600">{result.features.customerAge}</div>
                  <div className="text-xs text-gray-500 mt-1">years old</div>
                </div>
                
                <div className="p-4 bg-purple-50 rounded-xl">
                  <div className="text-xs text-gray-600 font-semibold mb-1">Vehicle Age</div>
                  <div className="text-2xl font-bold text-purple-600">{result.features.vehicleAge}</div>
                  <div className="text-xs text-gray-500 mt-1">years old</div>
                </div>
                
                <div className="p-4 bg-pink-50 rounded-xl">
                  <div className="text-xs text-gray-600 font-semibold mb-1">Subscription</div>
                  <div className="text-2xl font-bold text-pink-600">{result.features.subscription}</div>
                  <div className="text-xs text-gray-500 mt-1">months</div>
                </div>
                
                <div className="p-4 bg-orange-50 rounded-xl">
                  <div className="text-xs text-gray-600 font-semibold mb-1">Airbags</div>
                  <div className="text-2xl font-bold text-orange-600">{result.features.airbags}</div>
                  <div className="text-xs text-gray-500 mt-1">safety airbags</div>
                </div>
                
                <div className="p-4 bg-green-50 rounded-xl">
                  <div className="text-xs text-gray-600 font-semibold mb-1">ESC System</div>
                  <div className="text-2xl font-bold text-green-600">
                    {result.features.hasESC ? '✓' : '✗'}
                  </div>
                  <div className="text-xs text-gray-500 mt-1">{result.features.hasESC ? 'equipped' : 'not equipped'}</div>
                </div>
                
                <div className="p-4 bg-teal-50 rounded-xl">
                  <div className="text-xs text-gray-600 font-semibold mb-1">Location Type</div>
                  <div className="text-2xl font-bold text-teal-600">
                    {result.features.isUrban ? '🏙️' : '🌳'}
                  </div>
                  <div className="text-xs text-gray-500 mt-1">{result.features.isUrban ? 'urban' : 'rural'}</div>
                </div>
              </div>
            </div>

            {/* Customer-specific CTA */}
            {mode === 'customer' && (
              <div className="bg-gradient-to-r from-indigo-600 to-purple-600 rounded-2xl shadow-2xl p-8 text-white">
                <div className="text-center">
                  <h3 className="text-2xl font-bold mb-3">Ready to Apply?</h3>
                  <p className="text-indigo-100 mb-6">
                    {result.decision.tier === 'APPROVE' 
                      ? "Great news! You're likely to be approved. Start your application now to lock in these rates."
                      : result.decision.tier === 'MONITOR' || result.decision.tier === 'CONDITIONAL'
                      ? "You can still get coverage! Speak with our team to discuss your options and conditions."
                      : "While standard coverage isn't available right now, we have alternative products that might work for you."}
                  </p>
                  <div className="flex gap-4 justify-center">
                    <button className="px-8 py-4 bg-white text-indigo-600 rounded-xl font-bold hover:bg-indigo-50 transition-all shadow-lg">
                      {result.decision.tier === 'APPROVE' ? 'Start Application' : 'Speak to Agent'}
                    </button>
                    <button className="px-8 py-4 bg-indigo-700 text-white rounded-xl font-bold hover:bg-indigo-800 transition-all">
                      Get Quote via Email
                    </button>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Footer */}
      <div className="bg-white border-t border-gray-200 mt-12">
        <div className="max-w-7xl mx-auto px-6 py-8">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mb-6">
            <div>
              <h4 className="font-bold text-gray-800 mb-3">About UnderwriteGPT</h4>
              <p className="text-sm text-gray-600 leading-relaxed">
                Powered by validated risk models trained on 58,592 real insurance policies. 
                Our AI uses RAG architecture with FAISS vector search and Sentence Transformers.
              </p>
            </div>
            
            <div>
              <h4 className="font-bold text-gray-800 mb-3">Model Performance</h4>
              <ul className="text-sm text-gray-600 space-y-2">
                <li>✓ Base claim rate: 6.4%</li>
                <li>✓ Risk discrimination: 8.15% improvement</li>
                <li>✓ Search latency: <200ms</li>
                <li>✓ Confidence threshold: 60%+</li>
              </ul>
            </div>
            
            <div>
              <h4 className="font-bold text-gray-800 mb-3">Risk Model Weights</h4>
              <ul className="text-sm text-gray-600 space-y-2">
                <li>📅 Subscription: 50.7% (highest)</li>
                <li>👤 Driver: 14.3%</li>
                <li>📍 Region: 13.9%</li>
                <li>🚗 Vehicle: 12.3%</li>
                <li>🛡️ Safety: 8.8%</li>
              </ul>
            </div>
          </div>
          
          <div className="text-center pt-6 border-t border-gray-200">
            <p className="text-sm text-gray-500">
              <strong>UnderwriteGPT v3.0</strong> | RAG + Validated Risk Engineering | 
              <span className="text-indigo-600 ml-2">This system provides decision support. Final approvals require licensed underwriter review.</span>
            </p>
          </div>
        </div>
      </div>

      <style jsx>{`
        @keyframes fadeIn {
          from {
            opacity: 0;
            transform: translateY(20px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }
        
        .animate-fadeIn {
          animation: fadeIn 0.6s ease-out;
        }
        
        .line-clamp-2 {
          display: -webkit-box;
          -webkit-line-clamp: 2;
          -webkit-box-orient: vertical;
          overflow: hidden;
        }
      `}</style>
    </div>
  );
};

export default UnderwriteGPTEnhanced;