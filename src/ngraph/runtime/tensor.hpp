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

#include <memory>
#include <vector>

#include "ngraph/descriptor/layout/tensor_layout.hpp"
#include "ngraph/descriptor/tensor.hpp"
#include "ngraph/runtime/backend.hpp"
#include "ngraph/shape.hpp"
#include "ngraph/strides.hpp"
#include "ngraph/type/element_type.hpp"

namespace ngraph
{
    namespace descriptor
    {
        class Value;
    }

    namespace runtime
    {
        class NGRAPH_API Tensor
        {
        protected:
            Tensor(const std::shared_ptr<ngraph::descriptor::Tensor>& descriptor)
                : m_descriptor(descriptor)
                , m_stale(true)
            {
            }

        public:
            virtual ~Tensor() {}
            Tensor& operator=(const Tensor&) = default;

            /// \brief Get tensor shape
            /// \return const reference to a Shape
            virtual const ngraph::Shape& get_shape() const;

            /// \brief Get tensor partial shape
            /// \return const reference to a PartialShape
            const ngraph::PartialShape& get_partial_shape() const;

            /// \brief Get tensor strides
            /// \return Strides
            virtual ngraph::Strides get_strides() const;

            /// \brief Get tensor element type
            /// \return element::Type
            virtual const element::Type& get_element_type() const;

            /// \brief Get number of elements in the tensor
            /// \return number of elements in the tensor
            virtual size_t get_element_count() const;

            /// \brief Get the size in bytes of the tensor
            /// \return number of bytes in tensor's allocation
            virtual size_t get_size_in_bytes() const;

            /// \brief Get tensor's unique name
            /// \return tensor's name
            const std::string& get_name() const;

            /// \brief Get tensor layout
            /// \return tensor layout
            std::shared_ptr<descriptor::layout::TensorLayout> get_tensor_layout() const;

            /// \brief Set tensor layout
            /// \param layout Layout to set
            void set_tensor_layout(const std::shared_ptr<descriptor::layout::TensorLayout>& layout);

            /// \brief Get the stale value of the tensor. A tensor is stale if its data is
            /// changed.
            /// \return true if there is new data in this tensor
            bool get_stale() const;

            /// \brief Set the stale value of the tensor. A tensor is stale if its data is
            /// changed.
            void set_stale(bool val);

            /// \brief Indicate that the memory_pointer buffer contains new data
            virtual void indicate_write_buffer_updated() = 0;

            /// \brief Blocking call that waits for output data to be available
            virtual void wait_for_read_data() const = 0;

            /// \brief copy bytes directly from source to this tensor
            /// \param source The source tensor
            virtual void copy_from(const ngraph::runtime::Tensor& source) NGRAPH_DEPRECATED(
                "Allocate buf_ptr with size=get_size_in_bytes(), then use source.read(buf_ptr, "
                "size) followed by this->write(buf_ptr, size)");

        protected:
            std::shared_ptr<ngraph::descriptor::Tensor> m_descriptor;
            bool m_stale;
        };

        using TensorViewPtrs = std::vector<std::shared_ptr<Tensor>>;
    }
}
