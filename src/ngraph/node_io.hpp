
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

#include <atomic>
#include <cstring>
#include <deque>
#include <iostream>
#include <memory>
#include <set>
#include <string>
#include <tuple>
#include <typeindex>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "ngraph/autodiff/adjoints.hpp"
#include "ngraph/check.hpp"
#include "ngraph/coordinate.hpp"
#include "ngraph/deprecated.hpp"
#include "ngraph/descriptor/input.hpp"
#include "ngraph/descriptor/output.hpp"
#include "ngraph/descriptor/tensor.hpp"
#include "ngraph/op/util/attr_types.hpp"
#include "ngraph/op/util/op_annotations.hpp"
#include "ngraph/placement.hpp"
#include "ngraph/strides.hpp"
#include "ngraph/type.hpp"

namespace ngraph
{
    class NodeInput;
    class NodeOutput;
}

/// \brief A handle for one of a node's inputs.
class NGRAPH_API ngraph::NodeInput
{
public:
    /// \brief Constructs a NodeInput.
    /// \param node Pointer to the node for the input handle.
    /// \param index The index of the input.
    NodeInput(const Node* node, size_t index)
        : m_node(node)
        , m_index(index)
    {
    }

    /// \return A pointer to the node referenced by this input handle.
    Node* get_node() const { return const_cast<Node*>(m_node); }
    /// \return The index of the input referred to by this input handle.
    size_t get_index() const { return m_index; }
    /// \return The element type of the input referred to by this input handle.
    const element::Type& get_element_type() const
    {
        return m_node->get_input_element_type(m_index);
    }
    /// \return The shape of the input referred to by this input handle.
    const Shape& get_shape() const { return m_node->get_input_shape(m_index); }
    /// \return The partial shape of the input referred to by this input handle.
    const PartialShape& get_partial_shape() const
    {
        return m_node->get_input_partial_shape(m_index);
    }
    /// \return A handle to the output that is connected to this input.
    NodeOutput get_source_output() const;
    /// \return A reference to the tensor descriptor for this input.
    descriptor::Tensor& get_tensor() const
    {
        return m_node->m_inputs.at(m_index).get_output().get_tensor();
    }
    /// \return A shared pointer to the tensor descriptor for this input.
    std::shared_ptr<descriptor::Tensor> get_tensor_ptr() const
    {
        return m_node->m_inputs.at(m_index).get_output().get_tensor_ptr();
    }
    /// \return true if this input is relevant to its node's output shapes; else false.
    bool get_is_relevant_to_shapes() const
    {
        return m_node->m_inputs.at(m_index).get_is_relevant_to_shape();
    }
    /// \return true if this input is relevant to its node's output values; else false.
    bool get_is_relevant_to_values() const
    {
        return m_node->m_inputs.at(m_index).get_is_relevant_to_value();
    }

    /// \brief Replaces the source output of this input.
    /// \param new_source_output A handle for the output that will replace this input's source.
    void replace_source_output(const NodeOutput& new_source_output) const;

    bool operator==(const NodeInput& other) const
    {
        return m_node == other.m_node && m_index == other.m_index;
    }
    bool operator!=(const NodeInput& other) const { return !(*this == other); }
    bool operator<(const NodeInput& other) const
    {
        return m_node < other.m_node || (m_node == other.m_node && m_index < other.m_index);
    }
    bool operator>(const NodeInput& other) const
    {
        return m_node > other.m_node || (m_node == other.m_node && m_index > other.m_index);
    }
    bool operator<=(const NodeInput& other) const { return !(*this > other); }
    bool operator>=(const NodeInput& other) const { return !(*this < other); }
private:
    const Node* m_node;
    const size_t m_index;
};

/// \brief A handle for one of a node's outputs.
class NGRAPH_API ngraph::NodeOutput
{
public:
    /// \brief Constructs a NodeOutput.
    /// \param node A pointer to the node for the output handle.
    /// \param index The index of the output.
    NodeOutput(const Node* node, size_t index)
        : m_node(const_cast<Node*>(node)->shared_from_this())
        , m_index(index)
    {
    }

    /// \brief Constructs a NodeOutput.
    /// \param node A `shared_ptr` to the node for the output handle.
    /// \param index The index of the output.
    ///
    /// TODO: Make a plan to deprecate this.
    NodeOutput(const std::shared_ptr<const Node>& node, size_t index)
        : m_node(std::const_pointer_cast<Node>(node))
        , m_index(index)
    {
    }

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
    Node* get_node() const { return m_node.get(); }
    /// \return A `shared_ptr` to the node referred to by this output handle.
    ///
    /// TODO: Make a plan to deprecate this.
    std::shared_ptr<Node> get_node_shared_ptr() const { return m_node; }
    /// \return A useable shared pointer to this output. If index 0, the node,
    /// otherwise find or create a GOE.
    std::shared_ptr<Node> as_single_output_node(bool for_get_output_element = true) const
        NGRAPH_DEPRECATED("Transitional.")
    {
        return m_node->get_output_as_single_output_node(m_index, for_get_output_element);
    }

    /// \return The index of the output referred to by this output handle.
    size_t get_index() const { return m_index; }
    /// \return A reference to the tensor descriptor for this output.
    descriptor::Tensor& get_tensor() const { return m_node->m_outputs.at(m_index).get_tensor(); }
    /// \return A shared point to the tensor ptr for this output.
    std::shared_ptr<descriptor::Tensor> get_tensor_ptr() const
    {
        return m_node->m_outputs.at(m_index).get_tensor_ptr();
    }
    /// \return The element type of the output referred to by this output handle.
    const element::Type& get_element_type() const
    {
        return m_node->get_output_element_type(m_index);
    }
    /// \return The shape of the output referred to by this output handle.
    const Shape& get_shape() const { return m_node->get_output_shape(m_index); }
    /// \return The partial shape of the output referred to by this output handle.
    const PartialShape& get_partial_shape() const
    {
        return m_node->get_output_partial_shape(m_index);
    }

    /// \return A set containing handles for all inputs targeted by the output referenced by
    ///        this output handle.
    std::set<NodeInput> get_target_inputs() const;

    /// \brief Removes a target input from the output referenced by this output handle.
    /// \param target_input The target input to remove.
    ///
    // TODO(amprocte): Investigate whether this really ought to be public.
    void remove_target_input(const NodeInput& target_input) const;

    bool operator==(const NodeOutput& other) const
    {
        return m_node == other.m_node && m_index == other.m_index;
    }
    bool operator!=(const NodeOutput& other) const { return !(*this == other); }
    bool operator<(const NodeOutput& other) const
    {
        return m_node < other.m_node || (m_node == other.m_node && m_index < other.m_index);
    }
    bool operator>(const NodeOutput& other) const
    {
        return m_node > other.m_node || (m_node == other.m_node && m_index > other.m_index);
    }
    bool operator<=(const NodeOutput& other) const { return !(*this > other); }
    bool operator>=(const NodeOutput& other) const { return !(*this < other); }
private:
    std::shared_ptr<Node> m_node;
    size_t m_index{0};
};
