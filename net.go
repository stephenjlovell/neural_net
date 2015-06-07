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

type Net struct {
  layers []*Layer

}

type Topology []uint
type Data     []float64

func (net *Net) FeedForward(d Data) {
  input_layer := net.layers[0]
  for i, neuron := range *input_layer {
    neuron.output = d[i]
  }
  for _, layer := range net.layers {
    for _, neuron := range *layer {
      neuron.FeedForward()
    }
  }
}

func (net *Net) BackPropegate(d Data) {
  
}

func (net *Net) CalculateResults() Data {
  return nil
}



func NewNet(t Topology) *Net {
  layer_count := len(t)
  assert(layer_count >= 3, "Neural net must include at least one input, output, and hidden layer.")
  net := &Net{
    layers: make([]*Layer, layer_count, layer_count),
  }
  net.layers[0] = NewLayer(0, t[0], t[1])
  for i := 1; i < layer_count-1; i++ {
    net.layers[i] = NewLayer(t[i-1], t[i], t[i+1])
  }
  net.layers[layer_count-1] = NewLayer(t[layer_count-2], t[layer_count-1], 0)

  for i := layer_count-1; i > 0; i-- {
    receivers, senders := net.layers[i], net.layers[i-1]
    for r, receiver := range *receivers {
      for _, sender := range *senders {
        sender.connections[r].out = receiver.in
      }
    }
  }

  return net
}









