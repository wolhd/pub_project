#include <google/protobuf/util/json_util.h>
#include "det.pb.h" // Replace with your generated Protobuf header
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <iomanip>
// ...
std::vector<std::string> split(const std::string& s, char delimiter) {
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream tokenStream(s);
    while (std::getline(tokenStream, token, delimiter)) {
        // Simple trim whitespace from beginning and end (optional but good practice)
        size_t first = token.find_first_not_of(" \t\r\n");
        if (std::string::npos == first) {
            tokens.push_back(""); // Handle empty field
            continue;
        }
        size_t last = token.find_last_not_of(" \t\r\n");
        tokens.push_back(token.substr(first, (last - first + 1)));
    }
    return tokens;
}
/**
 * @brief Converts CSV data from an input stream into a JSON string format.
 * @param is The input stream (e.g., std::ifstream or std::stringstream) containing CSV data.
 * @return The generated JSON string.
 */
std::vector<std::string> csvToJson(std::istream& is) {
    std::string line;
    std::vector<std::string> jsonLines;
    
    // 1. Read Headers
    if (!std::getline(is, line)) {
        return jsonLines; // Empty CSV
    }
    std::vector<std::string> headers = split(line, ',');
    
    
    bool isFirstRow = true;
    
    // 2. Process Data Rows
    while (std::getline(is, line)) {
        if (line.empty() || line.find_first_not_of(" \t\r\n") == std::string::npos) {
            continue; // Skip empty lines
        }
        
        std::vector<std::string> values = split(line, ',');
        
        // Ensure the number of values matches the number of headers
        if (values.size() != headers.size()) {
            std::cerr << "Warning: Skipping row with inconsistent column count: " << line << std::endl;
            continue;
        }
        
        // Add comma separator for previous objects
/*        if (!isFirstRow) {
            jsonStream << ",\n";
        }
        isFirstRow = false;
*/        
	std::ostringstream jsonStream;
        // 3. Construct JSON Object for the row
        jsonStream << "  {\n";
        
        for (size_t i = 0; i < headers.size(); ++i) {
            // Write key-value pair: "Header": "Value"
            jsonStream << "    \"" << headers[i] << "\": ";
            
            // Check if value is numeric (simple check for integer or float)
            bool isNumeric = values[i].find_first_not_of("0123456789.-") == std::string::npos && 
                             !values[i].empty() && 
                             !(values[i] == "." || values[i] == "-");
            
            if (isNumeric) {
                // Numeric values are typically stored without quotes in JSON
                jsonStream << values[i];
            } else {
                // String values must be quoted and escaped (simple escaping for demonstration)
                // In a real application, you would need to escape double quotes and backslashes.
                jsonStream << "\"" << values[i] << "\"";
            }
            
            // Add comma unless it's the last element in the object
            if (i < headers.size() - 1) {
                jsonStream << ",";
            }
            jsonStream << "\n";
        }
        
        jsonStream << "  }";
        jsonLines.push_back(jsonStream.str());
    }
    
    return jsonLines;
}
void jsonToProto(std::string json_string) {

Detection message; // Replace with your actual message type

google::protobuf::util::JsonParseOptions parse_options;
// Configure parse_options if needed (e.g., ignore unknown fields)

google::protobuf::util::Status status = google::protobuf::util::JsonStringToMessage(json_string, &message, parse_options);

//message.set_time( message.time() + 1111.1 );
if (status.ok()) {
    // Message successfully parsed
    // You can now access message.field1(), message.field2(), etc.
    std::cout << "print msg: " << std::endl << message.DebugString() << std::endl;
} else {
    // Error parsing JSON
    std::cerr << "Error parsing JSON: " << status.ToString() << std::endl;
}

}
void testCsvToJson() {
    std::string csvData = 
        "Name,Age,Active\n"
        "Alice,30,true\n"
        "Bob Smith,24,false\n"
        "Charlie,,true\n"
        "David Jones,67,true\n";
    std::string csvDataDet = 
        "name,id,time,secIds\n"
        "nameA,2,123.45,9\n"
        "nameB,3,124.45,8\n"
        "nameA,4,125.46,7\n";
        
    
    std::stringstream csvStream(csvDataDet);
    std::vector<std::string> jsonLines = csvToJson(csvStream);
    for ( auto json : jsonLines ) {
    	std::cout<<"json result: " << std::endl << json << std::endl;
    }
    for ( auto json : jsonLines ) {
    	jsonToProto( json );
    }
}

void testJson() {
//std::string json_string = R"({"name": "nameA", "id": "44", "time": "1234.0542", "secIds": [9,8] })";
std::string json_string = R"({"name": "nameA", "id": "44", "time": "1234.0542", "secIds": "5" })";

Detection message; // Replace with your actual message type

google::protobuf::util::JsonParseOptions parse_options;
// Configure parse_options if needed (e.g., ignore unknown fields)

google::protobuf::util::Status status = google::protobuf::util::JsonStringToMessage(json_string, &message, parse_options);

message.set_time( message.time() + 1111.1 );
if (status.ok()) {
    // Message successfully parsed
    // You can now access message.field1(), message.field2(), etc.
    std::cout << "print msg: " << std::endl << message.DebugString() << std::endl;
} else {
    // Error parsing JSON
    std::cerr << "Error parsing JSON: " << status.ToString() << std::endl;
}
}

int main() {
testCsvToJson();
return 0;
}

// g++ det.pb.cc test.cpp -lprotobuf

/*
syntax = "proto3"; // Specifies the Protobuf language version

message Detection {
  string name = 1;
  int32 id = 2;
  double time = 3;
  repeated int32 secIds = 4;
}
*/
