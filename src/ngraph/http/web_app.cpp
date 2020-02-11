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

#include <iostream>

#include "ngraph/http/web_app.hpp"

using namespace std;

static string master_page = R"(
    <html>
    <head>
        <title>Aeon Debug</title>
        <script src="//ajax.googleapis.com/ajax/libs/jquery/1.11.0/jquery.min.js"></script>
        <script src="//netdna.bootstrapcdn.com/bootstrap/3.1.1/js/bootstrap.min.js"></script>
        <link rel="stylesheet" type="text/css" href="//netdna.bootstrapcdn.com/bootstrap/3.1.1/css/bootstrap.min.css" />
    </head>
    <body>
        <!-- Static navbar -->
        <nav class="navbar navbar-default navbar-static-top">
          <div class="container">
            <div class="navbar-header">
              <button type="button" class="navbar-toggle collapsed" data-toggle="collapse" data-target="#navbar" aria-expanded="false" aria-controls="navbar">
                <span class="sr-only">Toggle navigation</span>
                <span class="icon-bar"></span>
                <span class="icon-bar"></span>
                <span class="icon-bar"></span>
              </button>
              <a class="navbar-brand" href="https://github.com/NervanaSystems/aeon">Aeon</a>
            </div>
            <div id="navbar" class="navbar-collapse collapse">
              <ul class="nav navbar-nav">
                <li><a href="/">Home</a></li>
                <li class="dropdown">
                  <a href="#" class="dropdown-toggle" data-toggle="dropdown" role="button" aria-haspopup="true" aria-expanded="false">Aeon Stats <span class="caret"></span></a>
                  <ul class="dropdown-menu">
                    <li><a href="/loader">Loader</a></li>
                  </ul>
                </li>
              </ul>
            </div><!--/.nav-collapse -->
          </div>
        </nav>

    <div class="container">
        $content
    </div> <!-- /container -->

    </body>
    </html>
)";

WebApp::WebApp(uint16_t port)
{
    page_request_handler fn =
        bind(&WebApp::process_page_request, this, placeholders::_1, placeholders::_2);
    web_server.register_page_handler(fn);
    web_server.start(port);
}

WebApp::~WebApp()
{
    // web_server.stop();
}

void WebApp::home_page(web::Page& p)
{
    time_t t = time(0);
    struct tm* now = localtime(&t);
    ostream& out = p.output_stream();

    out << "<span>Current time: " << asctime(now) << "</span>\n";

    out << "<table class=\"table table-striped\">\n";
    out << "  <thead>\n";
    out << "    <th>Name</th>\n";
    out << "    <th>State</th>\n";
    out << "  </thead>\n";
    out << "  <tbody>\n";
    // for (auto info : ngraph::async_manager_status)
    // {
    //     out << "<tr>";
    //     out << "<td> " << info->get_name() << "</td>";
    //     out << "<td>";
    //     switch (info->get_state())
    //     {
    //     case ngraph::async_state::idle: out << "idle"; break;
    //     case ngraph::async_state::wait_for_buffer: out << "waiting for buffer"; break;
    //     case ngraph::async_state::fetching_data: out << "fetching data"; break;
    //     case ngraph::async_state::processing: out << "processing"; break;
    //     }
    //     out << "</td>";
    //     out << "</tr>";
    // }
    out << "  </tbody>\n";
    out << "</table>\n";
}

void WebApp::stopwatch(web::Page& p)
{
}

void WebApp::loader(web::Page& p)
{
    ostream& out = p.output_stream();

    // for (ngraph::loader* current_loader : m_loader_list)
    // {
    //     auto config = current_loader->get_current_config();
    //     out << "<pre>";
    //     out << config.dump(4);
    //     out << "</pre>";

    //     // Fetch next output buffer
    //     const ngraph::fixed_buffer_map& fixed_buffer = *(current_loader->get_current_iter());
    //     const ngraph::buffer_fixed_size_elements* buffer_ptr = fixed_buffer["image"];
    //     if (buffer_ptr)
    //     {
    //         // explicit copy the data
    //         ngraph::buffer_fixed_size_elements image_buffer{*buffer_ptr};
    //         out << "<div class=\"container\">";
    //         for (size_t i = 0; i < image_buffer.get_item_count(); i++)
    //         {
    //             cv::Mat         mat = image_buffer.get_item_as_mat(i);
    //             vector<uint8_t> encoded;
    //             imencode(".jpg", mat, encoded);
    //             vector<char> b64 =
    //                 ngraph::base64::encode((const char*)encoded.data(), encoded.size());
    //             out << "\n<img src=\"data:image/jpg;base64,";
    //             p.raw_send(b64.data(), b64.size());
    //             out << "\" style=\"padding-top:5px\"";
    //             out << "class=\"image col-lg-3\" ";
    //             out << "/>";
    //         }
    //         out << "</div>";
    //     }
    // }
}

void WebApp::page_404(web::Page& p)
{
    ostream& out = p.output_stream();
    out << "<div class=\"jumbotron>";
    out << "<h1>Page Not Found</h1>";
    out << "</div>";
}

void WebApp::process_page_request(web::Page& p, const string& url)
{
    ostream& out = p.output_stream();
    (void)out;
    if (url == "/")
    {
        auto mc = bind(&WebApp::home_page, this, placeholders::_1);
        p.master_page_string(master_page, "$content", mc);
    }
    else if (url == "/stopwatch")
    {
        auto mc = bind(&WebApp::stopwatch, this, placeholders::_1);
        p.master_page_string(master_page, "$content", mc);
    }
    else if (url == "/loader")
    {
        auto mc = bind(&WebApp::loader, this, placeholders::_1);
        p.master_page_string(master_page, "$content", mc);
    }
    else
    {
        p.page_not_found();
    }
}
