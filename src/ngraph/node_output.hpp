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

#include "ngraph/deprecated.hpp"
#include "ngraph/ngraph_visibility.hpp"

namespace ngraph
{
    class Node;
    class NodeInput;
    class NodeOutput;
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

/// \brief A handle for one of a node's outputs.
class NGRAPH_API ngraph::NodeOutput
{
public:
    /// \brief Constructs a NodeOutput.
    /// \param node A pointer to the node for the output handle.
    /// \param index The index of the output.
    NodeOutput(const Node* node, size_t index);

    /// \brief Constructs a NodeOutput.
    /// \param node A `shared_ptr` to the node for the output handle.
    /// \param index The index of the output.
    ///
    /// TODO: Make a plan to deprecate this.
    NodeOutput(const std::shared_ptr<const Node>& node, size_t index);

    /// \brief Constructs a NodeOutput, referencing the zeroth output of the node.
    /// \param node A `shared_ptr` to the node for the output handle.
    template <typename T>
    NodeOutput(const std::shared_ptr<T>& node)
        : NodeOutput(node, 0)
    {
    }

    /// A null output
    NodeOutput() = default;

    /// This output position for a different node
    NodeOutput for_node(const std::shared_ptr<Node>& node) { return NodeOutput(node, m_index); }
    /// \return A pointer to the node referred to by this output handle.
    Node* get_node() const;
    /// \return A `shared_ptr` to the node referred to by this output handle.
    ///
    /// TODO: Make a plan to deprecate this.
    std::shared_ptr<Node> get_node_shared_ptr() const { return m_node; }
    /// \return A useable shared pointer to this output. If index 0, the node,
    /// otherwise find or create a GOE.
    std::shared_ptr<Node> as_single_output_node(bool for_get_output_element = true) const
        NGRAPH_DEPRECATED("Transitional.");

    /// \return The index of the output referred to by this output handle.
    size_t get_index() const { return m_index; }
    /// \return A reference to the tensor descriptor for this output.
    descriptor::Tensor& get_tensor() const;
    /// \return A shared point to the tensor ptr for this output.
    std::shared_ptr<descriptor::Tensor> get_tensor_ptr() const;
    /// \return The element type of the output referred to by this output handle.
    const element::Type& get_element_type() const;
    /// \return The shape of the output referred to by this output handle.
    const Shape& get_shape() const;
    /// \return The partial shape of the output referred to by this output handle.
    const PartialShape& get_partial_shape() const;

    /// \return A set containing handles for all inputs targeted by the output referenced by
    ///        this output handle.
    std::set<NodeInput> get_target_inputs() const;

    /// \brief Removes a target input from the output referenced by this output handle.
    /// \param target_input The target input to remove.
    ///
    // TODO(amprocte): Investigate whether this really ought to be public.
    void remove_target_input(const NodeInput& target_input) const;

    bool operator==(const NodeOutput& other) const;
    bool operator!=(const NodeOutput& other) const { return !(*this == other); }
    bool operator<(const NodeOutput& other) const;
    bool operator>(const NodeOutput& other) const;
    bool operator<=(const NodeOutput& other) const { return !(*this > other); }
    bool operator>=(const NodeOutput& other) const { return !(*this < other); }
private:
    std::shared_ptr<Node> m_node;
    size_t m_index{0};
};
