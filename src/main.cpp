#include <Arduino.h>
#include <TensorFlowLite.h>
#include <model_quantized.h>
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"
namespace {
  tflite::ErrorReporter* error_reporter = nullptr;
  const tflite::Model* model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* input = nullptr;
  constexpr int kTensorArenaSize = 200 * 1024;
  static uint8_t tensor_arena[kTensorArenaSize];
}
#define INPUT_BUFFER_SIZE 64
#define OUTPUT_BUFFER_SIZE 64
#define INT_ARRAY_SIZE 8

// put function declarations here:
int string_to_array(char *in_str, int *int_array);
void print_int_array(int *int_array, int array_len);



char received_char = (char)NULL;              
int chars_avail = 0;                    // input present on terminal
char out_str_buff[OUTPUT_BUFFER_SIZE];  // strings to print to terminal
char in_str_buff[INPUT_BUFFER_SIZE];    // stores input from terminal
int input_array[INT_ARRAY_SIZE];        // array of integers input by user

int in_buff_idx=0; // tracks current input location in input buffer
int array_length=0;
int array_sum=0;

void setup() {
  Serial.begin(115200);
  
  
  // Set up logging
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Map the model into a usable data structure
  model = tflite::GetModel(model_quantized_tflite);
  if (model == nullptr) {
    TF_LITE_REPORT_ERROR(error_reporter, "Failed to load the model.");
    return;
  }

  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Model provided is schema version %d, not supported version %d.",
                         model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  static tflite::MicroMutableOpResolver<5> micro_op_resolver;
  micro_op_resolver.AddAveragePool2D();
  micro_op_resolver.AddConv2D();
  micro_op_resolver.AddDepthwiseConv2D();
  micro_op_resolver.AddReshape();
  micro_op_resolver.AddSoftmax();

  static tflite::MicroInterpreter static_interpreter(
    model, micro_op_resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed. Error code: %d", allocate_status);
    return;
  }

  input = interpreter->input(0);

  // Verify the input tensor shape, it should match the model's expected shape
  Serial.print("Input tensor shape: ");
  for (int i = 0; i < input->dims->size; i++) {
    Serial.print(input->dims->data[i]);
    Serial.print(" ");
  }
  Serial.println();
  
}
void loop() {
  unsigned long t0, t1, t2;
  chars_avail = Serial.available(); 
  if (chars_avail > 0) {
    received_char = Serial.read();  // get the typed character and 
    Serial.print(received_char);    // echo to the terminal

    in_str_buff[in_buff_idx++] = received_char;  // add it to the buffer
    if (received_char == 13) {  // 13 decimal = newline character
      // user hit 'enter', so we'll process the line.
      Serial.print("About to process line: ");
      Serial.println(in_str_buff);

      // Process and print out the array
      array_length = string_to_array(in_str_buff, input_array);
      sprintf(out_str_buff, "Read in  %d integers: ", array_length);
      Serial.print(out_str_buff);
      print_int_array(input_array, array_length);
      // Check that there are exactly 7 integers entered
      if (array_length != 7) {
        Serial.println("Warning: You must enter exactly 7 integers.");
      } else {
        // Print out the array values for debugging
        t0 = micros();  // Start time for printing
  
        // Print statement
        Serial.println("test statement");
        t1 = micros();  // Time after printing
        Serial.print("Parsed integers: ");
        for (int i = 0; i < array_length; i++) {
          Serial.print(input_array[i]);
          Serial.print(" ");
        }
        Serial.println();
        for (int i = 0; i < array_length; i++) {
          input_array[i] = input_array[i] / 100.0f;  // Example scaling
        }
        TfLiteTensor* input_tensor = interpreter->input(0);

        // Handle input tensor type accordingly
        if (input_tensor->type == kTfLiteInt8) {
          Serial.println("Input tensor type: INT8");

          // Proper scaling of float input data to fit in the INT8 range [-128, 127]
          for (int i = 0; i < 7; i++) {
            int8_t quantized_value = (int8_t)((input_array[i] - 0.0f) * (255.0f / 6.0f) - 128.0f);  // Scale to [-128, 127]
            input_tensor->data.int8[i] = quantized_value;
          }
        }
        // Get the output tensor from the interpreter
        TfLiteTensor* output_tensor = interpreter->output(0);
        if (output_tensor->type == kTfLiteInt8) {
          int8_t output_value = output_tensor->data.int8[0];  // Access the output value (int8)
          output_value = output_value / 32.0f; // or another scale factor based on training
          Serial.print("Prediction result (int8): ");
          Serial.println(output_value);
          t2 = micros();  // Time after inference
        }
        unsigned long t_print = t1 - t0;
        unsigned long t_infer = t2 - t1;

        // Print the time results
        Serial.print("Printing time = ");
        Serial.print(t_print);
        Serial.print(" us.  Inference time = ");
        Serial.print(t_infer);
        Serial.println(" us.");
      }

      // Now clear the input buffer and reset the index to 0
      memset(in_str_buff, (char)0, INPUT_BUFFER_SIZE * sizeof(char)); 
      in_buff_idx = 0;
    } else if (in_buff_idx >= INPUT_BUFFER_SIZE) {
      memset(in_str_buff, (char)0, INPUT_BUFFER_SIZE * sizeof(char)); 
      in_buff_idx = 0;
    }
  }
}

int string_to_array(char *in_str, int *int_array) {
  int num_integers=0;
  char *token = strtok(in_str, ",");
  
  while (token != NULL) {
    int_array[num_integers++] = atoi(token);
    token = strtok(NULL, ",");
    if (num_integers >= INT_ARRAY_SIZE) {
      break;
    }
  }
  
  return num_integers;
}

void print_int_array(int *int_array, int array_len) {
  int curr_pos = 0; // track where in the output buffer we're writing

  sprintf(out_str_buff, "Integers: [");
  curr_pos = strlen(out_str_buff); // so the next write adds to the end
  for(int i=0;i<array_len;i++) {
    // sprintf returns number of char's written. use it to update current position
    curr_pos += sprintf(out_str_buff+curr_pos, "%d, ", int_array[i]);
  }
  sprintf(out_str_buff+curr_pos, "]\r\n");
  Serial.print(out_str_buff);
}

