#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cstdlib>

using namespace std;

template<typename T>
string join(const vector<T>& vec, const string& delimiter) {
    /**
     * Join elements of a vector into a single string with a specified delimiter.
     * @param vec The vector containing elements to join.
     * @param delimiter The string to insert between elements.
     * @return A single string with all elements joined by the delimiter.
     */
    ostringstream oss;
    for (size_t i = 0; i < vec.size(); ++i) {
        if (i != 0) oss << delimiter;
        oss << vec[i];
    }
    return oss.str();
};

vector<double> parse_sample_file(const string& filename) {
    /**
     * Parse a sample file and extract the sample (parameter) values.
     * @param filename The path to the sample file.
     * @return A vector containing the sample values.
     */
    vector<double> sample;
    ifstream file(filename);
    if (file.is_open()) {
        string line;
        if (getline(file, line)) {
            istringstream iss(line);
            string val;
            // Read whitespace-separated values
            while (iss >> val) {
                sample.push_back(stod(val));
            }
        }
        file.close();
    }
    return sample;
}

int main(int argc, char* argv[]) {
    // Default values
    int numticks = 289;
    string inputfile = "config.txt";
    double wxw = 0.6, wyw = 0.6, wzw = 0.6;

    // Parse arguments
    for (int i = 1; i < argc; ++i) {
        string arg = argv[i];
        if (arg == "--numticks" && i + 1 < argc) {
            numticks = atoi(argv[++i]);
        } else if (arg == "--inputfile" && i + 1 < argc) {
            inputfile = argv[++i];
        } else if (arg == "--wxw" && i + 1 < argc) {
            wxw = atof(argv[++i]);
        } else if (arg == "--wyw" && i + 1 < argc) {
            wyw = atof(argv[++i]);
        } else if (arg == "--wzw" && i + 1 < argc) {
            wzw = atof(argv[++i]);
        }
    }

    // Read parameters from Sample.txt
    vector<double> params = parse_sample_file("Sample.txt");
    
    // Target values from experimental data (GH10, GH2, GH5 for 72h and 144h)
    // Cell targets: [90, 90, 130, 87, 65, 85]
    // Collagen targets: [647000, 428000, 551000, 1000000, 912000, 649000]
    vector<double> target_cells = {90, 90, 130, 87, 65, 85};
    vector<double> target_collagen = {647000, 428000, 551000, 1000000, 912000, 649000};
    
    // Use first 5 parameters to control the outputs
    // Optimal parameters should be close to 1.0 for each target
    double p0 = params.size() > 0 ? params[0] : 1.0;
    double p1 = params.size() > 1 ? params[1] : 1.0;
    double p2 = params.size() > 2 ? params[2] : 1.0;
    double p3 = params.size() > 3 ? params[3] : 1.0;
    double p4 = params.size() > 4 ? params[4] : 1.0;
    
    // Determine which config we're simulating based on input file
    string config_type = "GH2"; // default
    if (inputfile.find("GH10") != string::npos || inputfile.find("gh10") != string::npos) {
        config_type = "GH10";
    } else if (inputfile.find("GH5") != string::npos || inputfile.find("gh5") != string::npos) {
        config_type = "GH5";
    }

    ofstream outfile("output/Output_Biomarkers.csv");
    vector<string> headers = {
        "clock","TNF","TGF","FGF","IL6","IL8","IL10","Tropocollagen","Collagen","FragentedCollagen",
        "Tropoelastin","Elastin","FragmentedElastin","HA","FragmentedHA","Damage","ActivatedFibroblast",
        "Fibroblast","Elastic Mod (Pa)","Swelling Ratio","Mass Loss (%)"
    };
    outfile << join(headers, ",") << "\n";

    for (int tick = 0; tick < numticks; ++tick) {
        // Calculate cell counts and collagen based on config type and parameters
        double base_cells = 50;
        double base_collagen = 100000;
        double total_fibroblasts = base_cells;
        double collagen = base_collagen;

        // Simple approach: use parameters as multipliers
        // When all parameters are 1.0, we get exact target values
        // The optimizer will converge to [1,1,1,1] to minimize error
        double param_multiplier = p0 * p1 * p2 * p3;

        if (tick == 144) {  // 72h
            if (config_type == "GH10") {
            total_fibroblasts = target_cells[0] * param_multiplier;
            collagen = target_collagen[0] * param_multiplier;
            } else if (config_type == "GH2") {
            total_fibroblasts = target_cells[2] * param_multiplier;
            collagen = target_collagen[2] * param_multiplier;
            } else if (config_type == "GH5") {
            total_fibroblasts = target_cells[4] * param_multiplier;
            collagen = target_collagen[4] * param_multiplier;
            }
        } else if (tick == 288) {  // 144h
            if (config_type == "GH10") {
            total_fibroblasts = target_cells[1] * param_multiplier;
            collagen = target_collagen[1] * param_multiplier;
            } else if (config_type == "GH2") {
            total_fibroblasts = target_cells[3] * param_multiplier;
            collagen = target_collagen[3] * param_multiplier;
            } else if (config_type == "GH5") {
            total_fibroblasts = target_cells[5] * param_multiplier;
            collagen = target_collagen[5] * param_multiplier;
            }
        } else {
            total_fibroblasts = 0;
            collagen = 0;
        }
        
        // Split total fibroblasts between activated and normal (60/40 split)
        double activated_fib = total_fibroblasts * 0.6;
        double normal_fib = total_fibroblasts * 0.4; 

        vector<double> row = {
            static_cast<double>(tick),
            0.1 * tick,          // TNF
            0.2 * tick,          // TGF
            0.3 * tick,          // FGF
            0.4 * tick,          // IL6
            0.5 * tick,          // IL8
            0.6 * tick,          // IL10
            1.0 * tick,          // Tropocollagen
            collagen,            // Collagen - controlled by parameters
            0.5 * tick,          // FragentedCollagen
            1.1 * tick,          // Tropoelastin
            2.1 * tick,          // Elastin
            0.6 * tick,          // FragmentedElastin
            0.7 * tick,          // HA
            0.8 * tick,          // FragmentedHA
            0.9 * tick,          // Damage
            activated_fib,       // ActivatedFibroblast - controlled by parameters
            normal_fib,          // Fibroblast - controlled by parameters
            1000.0 + 10 * tick,  // Elastic Mod (Pa)
            1.0 + 0.01 * tick,   // Swelling Ratio
            0.5 * tick           // Mass Loss (%)
        };
        outfile << join(row, ",") << "\n";
    }

    outfile.close();
    cout << "Mock simulation complete. Output_Biomarkers.csv generated.\n";
    return 0;
}