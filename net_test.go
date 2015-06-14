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
  "math/rand"
)

func test_transform(d float64) float64 {
  return d * d
}

func test_data() float64 {
  return rand.Float64()
}

func test_input(size int) Data {
  input := make(Data, size, size)
  for i, _ := range input {
    input[i] = test_data()
  }
  return input
}

func test_target(input Data) Data {
  l := len(input)
  target := make(Data, l, l)
  for i, d := range input {
    target[i] = test_transform(d)
  }
  return target
} 

// verify the net can run a basic test without error.
func TestNetSetup(t *testing.T) {
  l := uint(10)

  var topology = Topology{ l, l, l }

  input := test_input(10)
  target := test_target(input)

  for i := 0; i < len(input); i++ {
    target[i] = test_transform(input[i])
  }


  net := NewNet(topology)

  net.FeedForward(input)

  for i := 0; i < 1000; i++ {
    net.Backpropegate(target)    
  }


  results := net.GetResults()

  for i := uint(0); i < l; i++ {
    fmt.Printf("%.3f | %.3f | %.3f\n", input[i], target[i], results[i])
  }

}









