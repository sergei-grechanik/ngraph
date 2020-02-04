//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
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

#include <cstdio>

#include "ngraph/node.hpp"

namespace ngraph
{
    class Node;

    template <typename NodeType>
    class Input
    {
    };

    /// \brief A handle for one of a node's inputs.
    template <>
    class Input<Node>
    {
    public:
        /// \brief Constructs a Input.
        /// \param node Pointer to the node for the input handle.
        /// \param index The index of the input.
        Input(Node* node, size_t index)
            : m_node(node)
            , m_index(index)
        {
        }

        /// \return A pointer to the node referenced by this input handle.
        Node* get_node() const { return m_node; }
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
        Output<Node> get_source_output() const;
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
        void replace_source_output(const Output<Node>& new_source_output) const;

        bool operator==(const Input& other) const
        {
            return m_node == other.m_node && m_index == other.m_index;
        }
        bool operator!=(const Input& other) const { return !(*this == other); }
        bool operator<(const Input& other) const
        {
            return m_node < other.m_node || (m_node == other.m_node && m_index < other.m_index);
        }
        bool operator>(const Input& other) const
        {
            return m_node > other.m_node || (m_node == other.m_node && m_index > other.m_index);
        }
        bool operator<=(const Input& other) const { return !(*this > other); }
        bool operator>=(const Input& other) const { return !(*this < other); }
    private:
        Node* const m_node;
        const size_t m_index;
    };

    /// \brief A handle for one of a node's inputs.
    template <>
    class NGRAPH_API Input<const Node>
    {
    public:
        /// \brief Constructs a Input.
        /// \param node Pointer to the node for the input handle.
        /// \param index The index of the input.
        Input(const Node* node, size_t index)
            : m_node(node)
            , m_index(index)
        {
        }

        /// \return A pointer to the node referenced by this input handle.
        const Node* get_node() const { return m_node; }
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
        Output<Node> get_source_output() const;
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

        bool operator==(const Input& other) const
        {
            return m_node == other.m_node && m_index == other.m_index;
        }
        bool operator!=(const Input& other) const { return !(*this == other); }
        bool operator<(const Input& other) const
        {
            return m_node < other.m_node || (m_node == other.m_node && m_index < other.m_index);
        }
        bool operator>(const Input& other) const
        {
            return m_node > other.m_node || (m_node == other.m_node && m_index > other.m_index);
        }
        bool operator<=(const Input& other) const { return !(*this > other); }
        bool operator>=(const Input& other) const { return !(*this < other); }
    private:
        const Node* const m_node;
        const size_t m_index;
    };

    inline Output<Node> Input<Node>::get_source_output() const
    {
        auto& output_descriptor = m_node->m_inputs.at(m_index).get_output();
        return Output<Node>(output_descriptor.get_node(), output_descriptor.get_index());
    }

    inline Output<Node> Input<const Node>::get_source_output() const
    {
        auto& output_descriptor = m_node->m_inputs.at(m_index).get_output();
        return Output<Node>(output_descriptor.get_node(), output_descriptor.get_index());
    }

    inline void Input<Node>::replace_source_output(const Output<Node>& new_source_output) const
    {
        m_node->m_inputs.at(m_index).replace_output(new_source_output.get_node_shared_ptr(),
                                                    new_source_output.get_index());
    }

    NGRAPH_API std::ostream& operator<<(std::ostream& out, const Input<Node>& input);
    NGRAPH_API std::ostream& operator<<(std::ostream& out, const Input<const Node>& input);
}
