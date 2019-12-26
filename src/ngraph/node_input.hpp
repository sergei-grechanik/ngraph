//*****************************************************************************
// Copyright 2017-2019 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#pragma once

#include <cstddef>
#include <memory>
#include <set>

#include "ngraph/ngraph_visibility.hpp"

namespace ngraph
{
    class Node;
    class NodeInput;
    class NodeOutput;
    class Shape;
    class PartialShape;
    namespace descriptor
    {
        class Tensor;
    }
    namespace element
    {
        class Type;
    }
}

/// \brief A handle for one of a node's inputs.
class NGRAPH_API ngraph::NodeInput
{
public:
    /// \brief Constructs a NodeInput.
    /// \param node Pointer to the node for the input handle.
    /// \param index The index of the input.
    NodeInput(Node* node, size_t index);

    /// \return A pointer to the node referenced by this input handle.
    Node* get_node() const { return m_node; }
    /// \return The index of the input referred to by this input handle.
    size_t get_index() const { return m_index; }
    /// \return The element type of the input referred to by this input handle.
    const element::Type& get_element_type() const;
    /// \return The shape of the input referred to by this input handle.
    const Shape& get_shape() const;
    /// \return The partial shape of the input referred to by this input handle.
    const PartialShape& get_partial_shape() const;
    /// \return A handle to the output that is connected to this input.
    NodeOutput get_source_output() const;
    /// \return A reference to the tensor descriptor for this input.
    descriptor::Tensor& get_tensor() const;
    /// \return A shared pointer to the tensor descriptor for this input.
    std::shared_ptr<descriptor::Tensor> get_tensor_ptr() const;
    /// \return true if this input is relevant to its node's output shapes; else false.
    bool get_is_relevant_to_shapes() const;
    /// \return true if this input is relevant to its node's output values; else false.
    bool get_is_relevant_to_values() const;

    /// \brief Replaces the source output of this input.
    /// \param new_source_output A handle for the output that will replace this input's source.
    void replace_source_output(const NodeOutput& new_source_output) const;

    bool operator==(const NodeInput& other) const;
    bool operator!=(const NodeInput& other) const { return !(*this == other); }
    bool operator<(const NodeInput& other) const;
    bool operator>(const NodeInput& other) const;
    bool operator<=(const NodeInput& other) const { return !(*this > other); }
    bool operator>=(const NodeInput& other) const { return !(*this < other); }
private:
    Node* m_node;
    const size_t m_index;
};
