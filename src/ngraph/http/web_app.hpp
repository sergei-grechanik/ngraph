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

#include "ngraph/http/http_server.hpp"

class WebApp
{
public:
    WebApp(uint16_t port);
    ~WebApp();

    void home_page(web::Page& p);
    void stopwatch(web::Page& p);
    void loader(web::Page& p);
    void page_404(web::Page& p);
    void process_page_request(web::Page& p, const std::string& url);

private:
    web::Server web_server;
};
