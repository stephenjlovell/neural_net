//-----------------------------------------------------------------------------------
// ♛ GopherCheck ♛
// Copyright © 2015 Stephen J. Lovell
//-----------------------------------------------------------------------------------
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//-----------------------------------------------------------------------------------

package NeuralNet

import(
  "fmt"
  "testing"
  // "math"
  // "math/rand"
)

// verify the net can run a basic test without error.
func TestNetSetup(t *testing.T) {
  l := uint(50)

  var topology = Topology{ l, l, l, 1 }
  net := NewNet(topology)


  for run := 0; run <= 5000; run++ {

    input := test_input(l)
    target := test_target(input) // target values should be scaled to all lie within 
                                 // range of Neuron's activation function    
    net.FeedForward(input)

    net.Backpropegate(target)    

    results := net.GetResults()

    if run % 200 == 0 {

      fmt.Printf("\nRun %d Error: %.4f Avg.Error: %.4f\n", run, net.error, net.recent_avg_err)  
      fmt.Printf("Inputs: %.2v\n", input)
      fmt.Printf("Target: %.2v\n", target)
      fmt.Printf("Results: %.2v\n", results)
    }

  }





}


func test_data() float64 { // random value [-1.0, 1.0]
  return random_weight()
}

func test_transform(d float64) float64 {
  return d * d
}

func test_input(size uint) Data {
  input := make(Data, size, size)
  for i := uint(0); i < size; i++ {
    input[i] = test_data()
  }
  return input
}

func test_target(input Data) Data {
  target := make(Data, 1)
  sum := 0.0
  for _, d := range input {
    sum += d
  }
  target[0] = (sum / float64(len(input)))
  return target
} 











