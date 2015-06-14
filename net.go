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
  "math"
)

type Net struct {
  layers []*Layer

}

type Topology []uint
type Data     []float64

func (net *Net) FeedForward(input Data) {
  input_layer := net.layers[0]
  for i, neuron := range *input_layer {
    neuron.FeedInitial(input[i])
  }
  for _, layer := range net.layers[1:] {
    for _, neuron := range *layer {
      neuron.FeedForward()
    }
  }
}

func (net *Net) OutputLayer() Layer {
  return *net.layers[len(net.layers)-1]
}

func (net *Net) InputLayer() Layer {
  return *net.layers[0]
}

func (net *Net) Backpropegate(target Data) {
  // calculate net error (RMS)
  err := 0.0
  out_layer := net.OutputLayer()
  for i, neuron := range out_layer {
    delta := target[i] - neuron.Output()
    err += (delta * delta)
  }
  err = math.Sqrt(err/float64(len(target))) // RMS
  // calculate output layer gradients
  for i, neuron := range out_layer {
    neuron.outputGradient(target[i])
  }
  // calculate gradients on hidden layers
  for i := len(net.layers)-2; i > 0; i-- {
    layer, next_layer := net.layers[i], net.layers[i+1]
    for _, neuron := range *layer {
      neuron.calcHiddenGradients(next_layer)
    }
  }
  // update connection weights for all layers
  for i := len(net.layers)-1; i > 0; i-- {
    layer, prev_layer := net.layers[i], net.layers[i-1]
    for _, neuron := range *layer {
      neuron.updateInputWeights(prev_layer)
    }
  }
}

func (net *Net) GetResults() Data {
  output_layer := net.OutputLayer()
  l := len(output_layer)
  data := make(Data, l, l)
  for i, n := range output_layer {
    data[i] = n.output
  }
  return data
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









