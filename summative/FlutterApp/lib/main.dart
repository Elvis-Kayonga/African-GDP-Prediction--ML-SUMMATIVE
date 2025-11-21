import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'African GDP Growth Predictor',
      theme: ThemeData(
        primarySwatch: Colors.teal,
        useMaterial3: true,
        colorScheme: ColorScheme.fromSeed(
          seedColor: Colors.teal,
          brightness: Brightness.light,
        ),
      ),
      home: const PredictionScreen(),
      debugShowCheckedModeBanner: false,
    );
  }
}

class PredictionScreen extends StatefulWidget {
  const PredictionScreen({super.key});

  @override
  State<PredictionScreen> createState() => _PredictionScreenState();
}

class _PredictionScreenState extends State<PredictionScreen> {
  // Controllers for text fields
  final TextEditingController yearController = TextEditingController();
  final TextEditingController inflationController = TextEditingController();
  final TextEditingController unemploymentController = TextEditingController();
  final TextEditingController fdiController = TextEditingController();
  final TextEditingController tradeBalanceController = TextEditingController();
  final TextEditingController debtController = TextEditingController();
  final TextEditingController internetController = TextEditingController();
  
  // Selected country
  String? selectedCountry;
  
  // Available countries (all 54 African countries from dataset)
  List<String> countries = [
    'Algeria', 'Angola', 'Benin', 'Botswana', 'Burkina Faso', 'Burundi',
    'Cameroon', 'Cape Verde', 'Central African Republic', 'Chad', 'Comoros',
    'Congo', 'DR Congo', 'Djibouti', 'Egypt', 'Equatorial Guinea', 'Eritrea',
    'Eswatini', 'Ethiopia', 'Gabon', 'Gambia', 'Ghana', 'Guinea', 'Guinea-Bissau',
    'Ivory Coast', 'Kenya', 'Lesotho', 'Liberia', 'Libya', 'Madagascar',
    'Malawi', 'Mali', 'Mauritania', 'Mauritius', 'Morocco', 'Mozambique',
    'Namibia', 'Niger', 'Nigeria', 'Rwanda', 'Sao Tome and Principe', 'Senegal',
    'Seychelles', 'Sierra Leone', 'Somalia', 'South Africa', 'South Sudan',
    'Sudan', 'Tanzania', 'Togo', 'Tunisia', 'Uganda', 'Zambia', 'Zimbabwe'
  ];
  
  // Prediction result
  String predictionResult = '';
  bool isLoading = false;
  bool hasError = false;
  
  // API URL - Render deployment
  final String apiUrl = 'https://african-gdp-api-qlax.onrender.com/predict';
  
  // Make prediction
  Future<void> makePrediction() async {
    // Validate inputs
    if (!validateInputs()) {
      return;
    }
    
    setState(() {
      isLoading = true;
      hasError = false;
      predictionResult = '';
    });
    
    try {
      // Prepare request body
      final Map<String, dynamic> requestBody = {
        'year': int.parse(yearController.text),
        'inflation_rate': double.parse(inflationController.text),
        'unemployment_rate': double.parse(unemploymentController.text),
        'fdi_millions_usd': double.parse(fdiController.text),
        'trade_balance_millions_usd': double.parse(tradeBalanceController.text),
        'govt_debt_percent_gdp': double.parse(debtController.text),
        'internet_penetration_percent': double.parse(internetController.text),
        'country': selectedCountry!,
      };
      
      // Make API request
      final response = await http.post(
        Uri.parse(apiUrl),
        headers: {'Content-Type': 'application/json'},
        body: json.encode(requestBody),
      );
      
      if (response.statusCode == 200) {
        // Success
        final data = json.decode(response.body);
        setState(() {
          predictionResult = 
              '✅ Predicted GDP Growth Rate:\n\n${data['predicted_gdp_growth_rate']}%\n\nModel: ${data['model_used']}';
          isLoading = false;
          hasError = false;
        });
      } else {
        // Error response
        final error = json.decode(response.body);
        setState(() {
          predictionResult = '❌ Error: ${error['detail'] ?? 'Unknown error'}';
          isLoading = false;
          hasError = true;
        });
      }
    } catch (e) {
      setState(() {
        predictionResult = '❌ Connection Error:\n\n${e.toString()}\n\nPlease check:\n- API URL is correct\n- Internet connection\n- API server is running';
        isLoading = false;
        hasError = true;
      });
    }
  }
  
  // Validate inputs
  bool validateInputs() {
    if (yearController.text.isEmpty ||
        inflationController.text.isEmpty ||
        unemploymentController.text.isEmpty ||
        fdiController.text.isEmpty ||
        tradeBalanceController.text.isEmpty ||
        debtController.text.isEmpty ||
        internetController.text.isEmpty ||
        selectedCountry == null) {
      setState(() {
        predictionResult = '❌ Error: Please fill in all fields';
        hasError = true;
      });
      return false;
    }
    
    // Validate numeric inputs
    try {
      int year = int.parse(yearController.text);
      double inflation = double.parse(inflationController.text);
      double unemployment = double.parse(unemploymentController.text);
      double fdi = double.parse(fdiController.text);
      double tradeBalance = double.parse(tradeBalanceController.text);
      double debt = double.parse(debtController.text);
      double internet = double.parse(internetController.text);
      
      // Range validation
      if (year < 2000 || year > 2050) {
        throw Exception('Year must be between 2000 and 2050');
      }
      if (inflation < 0 || inflation > 100) {
        throw Exception('Inflation rate must be between 0 and 100');
      }
      if (unemployment < 0 || unemployment > 100) {
        throw Exception('Unemployment rate must be between 0 and 100');
      }
      if (fdi < 0 || fdi > 50000) {
        throw Exception('FDI must be between 0 and 50000');
      }
      if (tradeBalance < -50000 || tradeBalance > 50000) {
        throw Exception('Trade balance must be between -50000 and 50000');
      }
      if (debt < 0 || debt > 200) {
        throw Exception('Government debt must be between 0 and 200');
      }
      if (internet < 0 || internet > 100) {
        throw Exception('Internet penetration must be between 0 and 100');
      }
      
      return true;
    } catch (e) {
      setState(() {
        predictionResult = '❌ Validation Error:\n\n${e.toString()}';
        hasError = true;
      });
      return false;
    }
  }
  
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text(
          'African GDP Growth Predictor',
          style: TextStyle(fontWeight: FontWeight.bold),
        ),
        centerTitle: true,
        elevation: 2,
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(20.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // Header
            const Card(
              elevation: 2,
              child: Padding(
                padding: EdgeInsets.all(16.0),
                child: Column(
                  children: [
                    Icon(Icons.analytics, size: 48, color: Colors.teal),
                    SizedBox(height: 8),
                    Text(
                      'Predict GDP Growth Rate',
                      style: TextStyle(
                        fontSize: 20,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                    SizedBox(height: 4),
                    Text(
                      'Enter economic indicators for African countries',
                      style: TextStyle(fontSize: 14, color: Colors.grey),
                      textAlign: TextAlign.center,
                    ),
                  ],
                ),
              ),
            ),
            
            const SizedBox(height: 24),
            
            // Country Dropdown
            DropdownButtonFormField<String>(
              value: selectedCountry,
              decoration: const InputDecoration(
                labelText: 'Country',
                border: OutlineInputBorder(),
                prefixIcon: Icon(Icons.flag),
              ),
              items: countries.map((country) {
                return DropdownMenuItem(
                  value: country,
                  child: Text(country),
                );
              }).toList(),
              onChanged: (value) {
                setState(() {
                  selectedCountry = value;
                });
              },
            ),
            
            const SizedBox(height: 16),
            
            // Year
            TextField(
              controller: yearController,
              decoration: const InputDecoration(
                labelText: 'Year (2000-2050)',
                border: OutlineInputBorder(),
                prefixIcon: Icon(Icons.calendar_today),
                hintText: 'e.g., 2024',
              ),
              keyboardType: TextInputType.number,
            ),
            
            const SizedBox(height: 16),
            
            // Inflation Rate
            TextField(
              controller: inflationController,
              decoration: const InputDecoration(
                labelText: 'Inflation Rate (%) (0-100)',
                border: OutlineInputBorder(),
                prefixIcon: Icon(Icons.trending_up),
                hintText: 'e.g., 12.5',
              ),
              keyboardType: const TextInputType.numberWithOptions(decimal: true),
            ),
            
            const SizedBox(height: 16),
            
            // Unemployment Rate
            TextField(
              controller: unemploymentController,
              decoration: const InputDecoration(
                labelText: 'Unemployment Rate (%) (0-100)',
                border: OutlineInputBorder(),
                prefixIcon: Icon(Icons.work_off),
                hintText: 'e.g., 18.0',
              ),
              keyboardType: const TextInputType.numberWithOptions(decimal: true),
            ),
            
            const SizedBox(height: 16),
            
            // FDI
            TextField(
              controller: fdiController,
              decoration: const InputDecoration(
                labelText: 'FDI (Millions USD) (0-50000)',
                border: OutlineInputBorder(),
                prefixIcon: Icon(Icons.attach_money),
                hintText: 'e.g., 3500',
              ),
              keyboardType: const TextInputType.numberWithOptions(decimal: true),
            ),
            
            const SizedBox(height: 16),
            
            // Trade Balance
            TextField(
              controller: tradeBalanceController,
              decoration: const InputDecoration(
                labelText: 'Trade Balance (Millions USD) (-50000 to 50000)',
                border: OutlineInputBorder(),
                prefixIcon: Icon(Icons.balance),
                hintText: 'e.g., -800',
              ),
              keyboardType: const TextInputType.numberWithOptions(decimal: true, signed: true),
            ),
            
            const SizedBox(height: 16),
            
            // Government Debt
            TextField(
              controller: debtController,
              decoration: const InputDecoration(
                labelText: 'Government Debt (% of GDP) (0-200)',
                border: OutlineInputBorder(),
                prefixIcon: Icon(Icons.account_balance),
                hintText: 'e.g., 38.0',
              ),
              keyboardType: const TextInputType.numberWithOptions(decimal: true),
            ),
            
            const SizedBox(height: 16),
            
            // Internet Penetration
            TextField(
              controller: internetController,
              decoration: const InputDecoration(
                labelText: 'Internet Penetration (%) (0-100)',
                border: OutlineInputBorder(),
                prefixIcon: Icon(Icons.wifi),
                hintText: 'e.g., 55.0',
              ),
              keyboardType: const TextInputType.numberWithOptions(decimal: true),
            ),
            
            const SizedBox(height: 24),
            
            // Predict Button
            ElevatedButton(
              onPressed: isLoading ? null : makePrediction,
              style: ElevatedButton.styleFrom(
                padding: const EdgeInsets.symmetric(vertical: 16),
                backgroundColor: Colors.teal,
                foregroundColor: Colors.white,
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(8),
                ),
              ),
              child: isLoading
                  ? const SizedBox(
                      height: 20,
                      width: 20,
                      child: CircularProgressIndicator(
                        color: Colors.white,
                        strokeWidth: 2,
                      ),
                    )
                  : const Text(
                      'Predict',
                      style: TextStyle(
                        fontSize: 18,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
            ),
            
            const SizedBox(height: 24),
            
            // Prediction Result
            if (predictionResult.isNotEmpty)
              Card(
                elevation: 3,
                color: hasError ? Colors.red.shade50 : Colors.green.shade50,
                child: Padding(
                  padding: const EdgeInsets.all(20.0),
                  child: Column(
                    children: [
                      Icon(
                        hasError ? Icons.error_outline : Icons.check_circle_outline,
                        size: 48,
                        color: hasError ? Colors.red : Colors.green,
                      ),
                      const SizedBox(height: 12),
                      Text(
                        predictionResult,
                        style: TextStyle(
                          fontSize: 16,
                          fontWeight: FontWeight.bold,
                          color: hasError ? Colors.red.shade900 : Colors.green.shade900,
                        ),
                        textAlign: TextAlign.center,
                      ),
                    ],
                  ),
                ),
              ),
          ],
        ),
      ),
    );
  }
  
  @override
  void dispose() {
    yearController.dispose();
    inflationController.dispose();
    unemploymentController.dispose();
    fdiController.dispose();
    tradeBalanceController.dispose();
    debtController.dispose();
    internetController.dispose();
    super.dispose();
  }
}
