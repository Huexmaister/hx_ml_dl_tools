from singleton_tools import SingletonMeta
import os

class HxModelEvaluationGraphics(metaclass=SingletonMeta):
    def __init__(self):
        pass

    @staticmethod
    def plot_dataframe_html(df, save_path: str, save: bool = False):
        # Convertir el DataFrame a una tabla HTML
        html_table = df.to_html(classes='display', index=True, border=0)

        # Crear el HTML completo incluyendo DataTable de JavaScript y estilos personalizados
        html = f"""
                <html>
                <head>
                    <link href="https://fonts.googleapis.com/css?family=Roboto:400,700&display=swap" rel="stylesheet">
                    <link rel="stylesheet" type="text/css" href="https://cdn.datatables.net/1.10.20/css/jquery.dataTables.css">
                    <script type="text/javascript" charset="utf8" src="https://code.jquery.com/jquery-3.3.1.min.js"></script>
                    <script type="text/javascript" charset="utf8" src="https://cdn.datatables.net/1.10.20/js/jquery.dataTables.js"></script>
                    <script>
                    $(document).ready( function () {{
                        var table = $('.display').DataTable({{
                            "scrollX": false,
                            "pageLength": 100,
                            "order": [],
                            "dom": '<"search-box"f>tip'
                        }});
                    }} );
                    </script>
                    <style>
                        body {{
                            font-family: 'Roboto', sans-serif;
                            margin: 20px;
                            padding: 20px;
                            background-color: #f5f5f7;
                        }}

                        .display {{
                            margin: 20px auto;
                            width: 90%;
                            border-collapse: collapse;
                            background-color: #fff;
                            border-radius: 10px;
                            box-shadow: 0 0 20px rgba(0, 0, 0, 0.1);
                        }}
                        th, td {{
                            padding: 15px;
                            text-align: left;
                            border-bottom: 1px solid #ddd;
                        }}
                        th {{
                            background-color: #333;
                            color: #fff;
                            font-weight: 700;
                            text-transform: uppercase;
                        }}
                        tr:nth-child(even) {{
                            background-color: #f5f5f7;
                        }}
                        tr:hover {{
                            background-color: #e9e9eb;
                        }}
                        .search-box-wrapper {{
                            text-align: center;
                            margin-top: 20px;
                            margin-bottom: 20px;
                        }}
                        .dataTables_filter_custom {{
                            display: block;
                            text-align: center;
                            margin-left: auto;
                            margin-right: auto;
                        }}
                        .dataTables_filter_custom input {{
                            width: 300px;
                            padding: 10px;
                            font-size: 16px;
                            border-radius: 5px;
                            border: 1px solid #ccc;
                        }}
                        .dataTables_wrapper .dataTables_filter {{
                            float: none !important;
                            text-align: center !important;
                            margin: 3%;
                        }}
                    </style>
                </head>
                <body>
                    {html_table}
                </body>
                </html>
                """

        if save:
            # Guardar el HTML
            with open(save_path, 'w') as f:
                f.write(html)

        else:
            # Guardar el HTML en un archivo temporal
            path = os.path.abspath('temp_table.html')
            with open(path, 'w') as f:
                f.write(html)

    @staticmethod
    def plot_dataframe_html_podium(df, save_path: str, save: bool = False):
        # Convertir el DataFrame a una tabla HTML
        html_table = df.to_html(classes='display', index=True, border=0)

        # Crear el HTML completo incluyendo DataTable de JavaScript y estilos personalizados
        html = f"""
                            <html>
                            <head>
                                <link href="https://fonts.googleapis.com/css?family=Roboto:400,700&display=swap" rel="stylesheet">
                                <link rel="stylesheet" type="text/css" href="https://cdn.datatables.net/1.10.20/css/jquery.dataTables.css">
                                <script type="text/javascript" charset="utf8" src="https://code.jquery.com/jquery-3.3.1.min.js"></script>
                                <script type="text/javascript" charset="utf8" src="https://cdn.datatables.net/1.10.20/js/jquery.dataTables.js"></script>
                                <script>
                                $(document).ready( function () {{
                                    var table = $('.display').DataTable({{
                                        "scrollX": false,
                                        "pageLength": 100,
                                        "order": [],
                                        "dom": '<"search-box"f>tip',
                                        "initComplete": function(settings, json) {{
                                            $(".search-box input").unwrap().wrap('<div class="dataTables_filter_custom"></div>');
                                            $(".dataTables_filter_custom").wrap('<div class="search-box-wrapper"></div>');
                                            // Resaltar las filas del índice 0, 1 y 2 para oro, plata y bronce
                                            $('.display tbody tr:eq(0)').addClass('gold');
                                            $('.display tbody tr:eq(1)').addClass('silver');
                                            $('.display tbody tr:eq(2)').addClass('bronze');
                                        }}
                                    }});
                                    table.on('order.dt', function () {{
                                        $('td').removeClass('column-selected');
                                        var order = table.order()[0];
                                        if (order) {{
                                            $('td:nth-child(' + (order[0]+1) + ')').addClass('column-selected');
                                        }}
                                    }});
                                }} );
                                </script>
                                <style>
                                    body {{
                                        font-family: 'Roboto', sans-serif;
                                        margin: 20px;
                                        padding: 20px;
                                        background-color: #f5f5f7;
                                    }}

                                    .display {{
                                        margin: 20px auto;
                                        width: 90%;
                                        border-collapse: collapse;
                                        background-color: #fff;
                                        border-radius: 10px;
                                        box-shadow: 0 0 20px rgba(0, 0, 0, 0.1);
                                    }}
                                    th, td {{
                                        padding: 15px;
                                        text-align: left;
                                        border-bottom: 1px solid #ddd;
                                    }}
                                    th {{
                                        background-color: #333;
                                        color: #fff;
                                        font-weight: 700;
                                        text-transform: uppercase;
                                    }}
                                    tr:nth-child(even) {{
                                        background-color: #f5f5f7;
                                    }}
                                    tr:hover {{
                                        background-color: #e9e9eb;
                                    }}
                                    .gold {{
                                        background-color: gold !important;
                                        font-weight: bold;
                                    }}
                                    .silver {{
                                        background-color: silver !important;
                                        font-weight: bold;
                                    }}
                                    .bronze {{
                                        background-color: #cd7f32 !important; /* Bronze color */
                                        font-weight: bold;
                                    }}
                                    .search-box-wrapper {{
                                        text-align: center;
                                        margin-top: 20px;
                                        margin-bottom: 20px;
                                    }}
                                    .dataTables_filter_custom {{
                                        display: block;
                                        text-align: center;
                                        margin-left: auto;
                                        margin-right: auto;
                                    }}
                                    .dataTables_filter_custom input {{
                                        width: 300px;
                                        padding: 10px;
                                        font-size: 16px;
                                        border-radius: 5px;
                                        border: 1px solid #ccc;
                                    }}
                                    .dataTables_wrapper .dataTables_filter {{
                                        float: none !important;
                                        text-align: center !important;
                                        margin: 3%;
                                    }}
                                </style>
                            </head>
                            <body>
                                {html_table}
                            </body>
                            </html>
                            """

        if save:
            # Guardar el HTML
            with open(save_path, 'w') as f:
                f.write(html)
        else:
            # Guardar el HTML en un archivo temporal
            path = os.path.abspath('temp_table.html')
            with open(path, 'w') as f:
                f.write(html)

