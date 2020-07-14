import os
import re
from collections import defaultdict

import altair as alt
import community
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.offline as py
import squarify
from matplotlib import pyplot, patches
from plotly.subplots import make_subplots
from scipy.sparse import dok_matrix
from sklearn import manifold
from sklearn.utils._joblib import Parallel, delayed

# for plotting
PATHWAY_TAG = 'Pathway ID'
PATHWAY_URL = 'https://metacyc.org/META/NEW-IMAGE?type=PATHWAY&object='
EC_TAG = 'EC ID'
EC_URL = 'https://metacyc.org/META/NEW-IMAGE?type=EC-NUMBER&object='
MOL_TAG = 'Mol ID'
MOL_URL = 'https://metacyc.org/compound?orgid=META&id='
ONTOLOGY_URL = 'https://metacyc.org/META/new-image?object='


def custom_colorscales(num_items, start_idx=190):
    idx = 0
    color_brewer = list()
    a = np.arange(start=start_idx, stop=256)
    while num_items > idx:
        r = np.random.choice(a=a, size=1)
        g = np.random.choice(a=a, size=1)
        b = np.random.choice(a=a, size=1)
        color = "rgb({0:s},{1:s},{2:s})".format(str(r[0]), str(g[0]), str(b[0]))
        if color in color_brewer:
            continue
        color_brewer.append(color)
        idx = idx + 1
    return color_brewer


def check_brightness(r, g, b):
    return (r * 299 + g * 587 + b * 114) / 1000


def custom_symbols(num_items):
    list_symbols = ['square', 'diamond', 'triangle-up', 'pentagon', 'hexagon', 'octagon', 'hexagram']
    symbols = np.random.choice(a=len(list_symbols), size=num_items)
    symbols = [list_symbols[i] for i in symbols]
    return symbols


def __check_validity(perplexity, num_epochs, num_labels2plot, min_node_size, min_num_nodes):
    if perplexity < 0:
        perplexity = 50
    if num_epochs < 0:
        num_epochs = 5000
    if num_labels2plot < 0:
        num_labels2plot = len([node[1]['pidx'] for node in hin.nodes(data=True)
                               if node[1]['type'] == "T"])
    if min_node_size < 10:
        min_node_size = 100
    if min_num_nodes < 0:
        min_num_nodes = 5
    return perplexity, num_epochs, num_labels2plot, min_node_size, min_num_nodes


def __extract_information(hin, pathway_info):
    # extracting pathway ontology information
    tmp = [(pathway[1][1][1], pathway[0]) for pathway in pathway_info.items()
           if pathway[0] in hin.nodes(data=True)]
    ontology_dict = dict()
    for t, pidx in tmp:
        for tid in t:
            if tid in ontology_dict:
                ontology_dict[tid] += [pidx]
            else:
                ontology_dict.update({tid: [pidx]})
    pathway_ontology_dict = dict((pathway[0], ', '.join(pathway[1][1][1])) for pathway
                                 in pathway_info.items() if pathway[0] in hin.nodes(data=True))
    # extracting species information
    tmp = [(pathway[1][8][1], pathway[0]) for pathway in pathway_info.items()
           if pathway[0] in hin.nodes(data=True)]
    species_dict = dict()
    for t, pidx in tmp:
        for tid in t:
            if tid in species_dict:
                species_dict[tid] += [pidx]
            else:
                species_dict.update({tid: [pidx]})
    pathway_species_dict = dict((pathway[0], ', '.join(pathway[1][8][1]))
                                for pathway in pathway_info.items()
                                if pathway[0] in hin.nodes(data=True))
    # extracting taxonomic range information
    tmp = [(pathway[1][9][1], pathway[0]) for pathway in pathway_info.items()
           if pathway[0] in hin.nodes(data=True)]
    taxonomic_dict = dict()
    for t, pidx in tmp:
        for tid in t:
            if tid in taxonomic_dict:
                taxonomic_dict[tid] += [pidx]
            else:
                taxonomic_dict.update({tid: [pidx]})
    pathway_taxonomic_dict = dict((pathway[0], ', '.join(pathway[1][9][1]))
                                  for pathway in pathway_info.items()
                                  if pathway[0] in hin.nodes(data=True))

    return ontology_dict, pathway_ontology_dict, species_dict, pathway_species_dict, taxonomic_dict, pathway_taxonomic_dict


def __get_dataframe(pathway_labels, species_dict, ontology_dict, pathway_species_dict, pathway_ontology_dict):
    species_df = pd.DataFrame(columns=['Pathway_ID', 'Species_ID', 'Species_Distribution'])
    for k, val in pathway_species_dict.items():
        if k in pathway_labels:
            tmp = val.split(',')
            tmp = [x.strip(' ') for x in tmp]
            for v in tmp:
                prior = 1 / len(tmp)
                ptwys = species_dict[v]
                sum_ptwys = len([p for p in ptwys if p in pathway_labels])
                post = prior * (sum_ptwys / len(ptwys))
                species_df = species_df.append({'Pathway_ID': k, 'Species_ID': v,
                                                'Species_Distribution': post},
                                               ignore_index=True)
    species_df = species_df.reset_index(drop=True)
    ontology_df = pd.DataFrame(columns=['Pathway_ID', 'Ontology_ID', 'Ontology_Distribution'])
    for k, val in pathway_ontology_dict.items():
        if k in pathway_labels:
            tmp = val.split(',')
            tmp = [x.strip(' ') for x in tmp]
            for v in tmp:
                prior = 1 / len(tmp)
                ptwys = ontology_dict[v]
                sum_ptwys = len([p for p in ptwys if p in pathway_labels])
                post = prior * (sum_ptwys / len(ptwys))
                ontology_df = ontology_df.append({'Pathway_ID': k, 'Ontology_ID': v,
                                                  'Ontology_Distribution': post},
                                                 ignore_index=True)
    ontology_df = ontology_df.reset_index(drop=True)
    merged_df = pd.merge(species_df, ontology_df, on=['Pathway_ID'])
    return merged_df, ontology_df, species_df


def __generate_network_object(G, pos=None, min_node_size=10, get_trace=False, disable_sizing=False,
                              node_symbol='circle-dot', node_colorscale='YlGnBu', colorbar_x=1.02,
                              xanchor='left', additional_dict=None, additional_dict_symbol=False,
                              additional_dict_tag=None, min_num_nodes=5, title=None, source=None,
                              hover_name=None):
    try:
        # generate a layout
        if pos is None:
            pos = nx.spring_layout(G=G)

        # whether to show scale for nodes
        showscale = False
        if not disable_sizing:
            showscale = True

        colors_dict = dict()
        symbols_dict = dict()
        if additional_dict:
            showscale = False
            colors_dict = dict(zip(additional_dict.keys(), custom_colorscales(len(additional_dict.keys()))))
            if additional_dict_symbol:
                symbols_dict = dict(zip(additional_dict.keys(), custom_symbols(len(additional_dict.keys()))))

        ## define the plotly graphical objects
        # first, add edge information
        edge_trace = go.Scatter(x=list(), y=list(), line=dict(width=1, color="rgb(180,180,180)"), hoverinfo='none',
                                mode='lines')
        sticky_text = list()
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_trace['x'] += (x0, x1, None)
            edge_trace['X'] += (y0, y1, None)
            if len(list(G.neighbors(edge[0]))) >= min_num_nodes:
                if additional_dict:
                    if edge[0] not in additional_dict.keys():
                        continue
                sticky_text.append(dict(text=edge[0],
                                        x=x0, y=y0,
                                        xref='x1', yref='y1',
                                        font=dict(color='rgb(50,50,50)', size=10),
                                        showarrow=False))
        # add node information
        colorbar = dict(x=colorbar_x, thickness=15, title='Node Connections', xanchor=xanchor, titleside='right')
        node_trace = go.Scatter(x=list(), y=list(), hoverinfo='text', mode='markers',
                                marker=dict(symbol=node_symbol, sizemode='area', showscale=showscale,
                                            colorscale=node_colorscale, reversescale=False, color=[],
                                            colorbar=colorbar, line=dict(color="rgb(50,50,50)", width=0.5)))

        tmp_x = list()
        tmp_y = list()
        for node in G.nodes():
            x, y = pos[node]
            tmp_x.append(x)
            tmp_y.append(y)
        node_trace['x'] = tuple(tmp_x)
        node_trace['X'] = tuple(tmp_y)

        # add adjacency and node size information
        node_symbols = []
        node_adjacencies = []
        node_size = []
        hover_text = []

        for adjacencies in G.adjacency():
            tag = hover_name
            if additional_dict:
                if adjacencies[0] in additional_dict.keys():
                    tag = additional_dict_tag
                    node_adjacencies.append(colors_dict[adjacencies[0]])
                    node_size.append(len(adjacencies[1]) * 100)
                    if additional_dict_symbol:
                        node_symbols.append(symbols_dict[adjacencies[0]])
                    else:
                        node_symbols.append("circle")
                else:
                    node_symbols.append("circle")
                    node_adjacencies.append("rgb(100,100,100)")
                    node_size.append(20)
            else:
                node_adjacencies.append(len(adjacencies[1]))
                node_size.append(len(adjacencies[1]) + min_node_size)
            hover_text.append(('<b>{name}</b>: {rid}<br>' +
                               '<b>Degree</b>: {degree}<br>').format(name=tag, rid=adjacencies[0],
                                                                     degree=len(adjacencies[1])))
        node_trace.text = hover_text
        node_trace.marker.symbol = node_symbol
        node_trace.marker.color = "LightBlue"
        node_trace.marker.size = len(G.nodes()) * [15]
        if not disable_sizing and additional_dict is None:
            sizeref = 2. * max(node_size) / (10 ** 2)
            node_trace.marker.color = node_adjacencies
            node_trace.marker.sizeref = sizeref
            node_trace.marker.size = node_size
        else:
            node_trace.marker.symbol = node_symbols
            node_trace.marker.color = node_adjacencies
            node_trace.marker.size = node_size

        if source is not None:
            annotations = [dict(showarrow=False, text=source, xref="paper", yref="paper",
                                x=0.005, y=-0.002, xanchor='left', yanchor='bottom',
                                font=dict(size=14))]
        else:
            annotations = []

        if sticky_text:
            annotations.extend(sticky_text)

        data = [edge_trace, node_trace]
        if get_trace:
            return data, annotations
        # define the plotly plot layout
        layout = go.Layout(title=title, titlefont_size=16, showlegend=False,
                           hovermode='closest', margin=dict(b=20, l=100, r=100, t=40),
                           annotations=annotations,
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        fig = go.Figure(data=data, layout=layout)
        return fig
    except ImportError as ex:
        print('Please install plotly...')
        print(ex)


def __generate_heatmap_object(G, dict_object, hover_name="Ontology ID", start_idx_color=0,
                              length_display_text=5, title="Pathway Ontology"):
    # extract texts and values
    values = list()
    sticky_text = list()
    hover_text = list()
    for node in dict_object.keys():
        if node in G.nodes():
            tmp = ['{0}'.format(n) for n in list(G.neighbors(node))]
            tmp = ', '.join(tmp)
            values.append(len(list(G.neighbors(node))))
            sticky_text.append(node)
            hover_text.append(('<b>{name}</b>: {text}<br>' +
                               '<b>#Pathways</b>: {degree}<br>' +
                               '<b>Pathways</b>: {item}<br>').format(name=hover_name,
                                                                     text=node,
                                                                     degree=len(list(G.neighbors(node))),
                                                                     ptwy=tmp))
    color_brewer = custom_colorscales(len(dict_object.keys()), start_idx=start_idx_color)

    # generate squares
    x = 0.
    y = 0.
    width = 100.
    height = 100.
    normed = squarify.normalize_sizes(values, width, height)
    rects = squarify.squarify(normed, x, y, width, height)

    # add shapes and annotations for layout
    shapes = list()
    annotations = list()
    counter = 0
    for rc, val, txt, clr in zip(rects, values, sticky_text, color_brewer):
        shapes.append(dict(type='rect', x0=rc['x'], y0=rc['X'],
                           x1=rc['x'] + rc['dx'], y1=rc['X'] + rc['dy'],
                           line=dict(width=1), fillcolor=clr))
        r, g, b = re.findall(r'\d+', clr)
        brightness = check_brightness(int(r), int(g), int(b))
        text_color = "rgb(0,0,0)"
        if (brightness < 123):
            text_color = "rgb(255,255,255)"

        annotations.append(dict(x=rc['x'] + (rc['dx'] / 2),
                                y=rc['X'] + (rc['dy'] / 2),
                                font=dict(color=text_color, size=12),
                                text=txt[:length_display_text] + '...',
                                showarrow=False))

    # for hover text
    plate_trace = go.Scatter(x=[r['x'] + (r['dx'] / 2) for r in rects],
                             y=[r['X'] + (r['dy'] / 2) for r in rects],
                             hoverinfo='text', mode='markers',
                             text=hover_text,
                             marker=dict(sizemode='area', showscale=False,
                                         color=color_brewer,
                                         line=dict(color='DarkSlateGrey', width=1)))

    # define the plotly plot layout
    layout = go.Layout(width=1000, height=600, title=title, titlefont_size=16, showlegend=False,
                       hovermode='closest', annotations=annotations, margin=dict(b=20, l=100, r=100, t=40),
                       xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       shapes=shapes, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    fig = go.Figure(data=plate_trace, layout=layout)
    return fig


def __generate_bar_object(df, x_val='Species_ID', y_val='Pathway_ID', distr_val="Species_Distribution",
                          get_trace=False, y_title='Species', title="Distribution of species by pathways"):
    tmp_id = list(set(list(df[x_val])))
    tmp_name = list(set(list(df[y_val])))
    id_distr = np.zeros((len(set(tmp_id)), len(set(tmp_name))))
    for ridx, name in enumerate(tmp_id):
        rows = df.loc[df[x_val] == name]
        for pid in rows[y_val]:
            pidx = tmp_name.index(pid)
            id_distr[ridx, pidx] = list(rows[distr_val])[0]

    # for adjusting input and hovering text
    data = list()
    for ridx, name in enumerate(tmp_name):
        trace = go.Bar(y=tmp_id, x=id_distr[:, ridx], name=tmp_name[ridx],
                       hovertemplate=
                       "<b>%{name}</b><br>" +
                       "<b>%{xaxis.title.text}</b>: %{x:0.4f}<br>" +
                       "<extra></extra>",
                       orientation='h', marker=dict(line=dict(color='DarkSlateGrey', width=1)))
        data.append(trace)

    if get_trace:
        return data

    # define the plotly plot layout
    layout = go.Layout(width=600, height=400, title=title, titlefont_size=16, showlegend=True,
                       hovermode='closest', margin=dict(b=20, l=100, r=100, t=40), barmode='stack',
                       xaxis=dict(title='Distribution', showgrid=False, zeroline=False,
                                  showticklabels=False),
                       yaxis=dict(title=y_title, showgrid=False, zeroline=False,
                                  showticklabels=True, categoryorder='category ascending'),
                       paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')

    fig = go.Figure(data=data, layout=layout)
    return fig


def __generate_pie_object(df, x_val='Ontology_ID', y_val='Pathway_ID', distr_val="Ontology_Distribution",
                          get_trace=False, title="Distribution of ontologies by pathways"):
    tmp_id = list(set(list(df[x_val])))
    tmp_name = list(set(list(df[y_val])))
    id_distr = np.zeros((len(set(tmp_id)), len(set(tmp_name))))
    for ridx, name in enumerate(tmp_id):
        rows = df.loc[df[x_val] == name]
        for pid in rows[y_val]:
            pidx = tmp_name.index(pid)
            id_distr[ridx, pidx] = list(rows[distr_val])[0]

    color_brewer = custom_colorscales(len(set(df[x_val])), start_idx=0)
    id_distr = np.sum(id_distr, axis=1)
    id_distr = id_distr / id_distr.sum()

    # for adjusting input and hovering text
    data = go.Pie(labels=tmp_id, values=id_distr, hoverinfo='label+percent',
                  textinfo='label', textfont_size=8,
                  marker=dict(colors=color_brewer,
                              line=dict(color='DarkSlateGrey',
                                        width=1)))
    if get_trace:
        return data

    # define the plotly plot layout
    layout = go.Layout(width=800, height=400, title=title, titlefont_size=16, showlegend=True,
                       hovermode='closest', margin=dict(b=20, l=100, r=100, t=40),
                       paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    fig = go.Figure(data=data, layout=layout)
    return fig


def visualize_layer(hin, labels, min_node_size, min_num_nodes, title="Pathway-Pathway Interaction Network",
                    hover_name=PATHWAY_TAG, filename="pathway.html", save_path='.'):
    G = nx.subgraph(G=hin, nbunch=labels)
    fig = __generate_network_object(G=G, min_node_size=min_node_size, get_trace=False, disable_sizing=False,
                                    node_colorscale='YlGnBu', node_symbol='circle-dot', colorbar_x=1.02, xanchor='left',
                                    additional_dict=None, additional_dict_symbol=False, additional_dict_tag=None,
                                    min_num_nodes=min_num_nodes, title=title, source=None, hover_name=hover_name)
    filename = os.path.join(save_path, filename)
    py.plot(fig, filename=filename, auto_open=False)


def visualize_species2pathways_heat(pathway_labels, species_dict, length_display_text, title="Pathway species heatmap",
                                    filename="pathway_species_heatmap.html", save_path='.'):
    G = nx.Graph()
    G.add_nodes_from(species_dict.keys())
    for key, val in species_dict.items():
        G.add_edges_from(([(key, t) for t in val if t in pathway_labels]))
    G.remove_nodes_from(list(nx.isolates(G)))
    fig = __generate_heatmap_object(G=G, dict_object=species_dict, hover_name="Species ID",
                                    start_idx_color=0, length_display_text=length_display_text,
                                    title=title)
    filename = os.path.join(save_path, filename)
    py.plot(fig, filename=filename, auto_open=False)


def visualize_pathway_ontology(pathway_labels, ontology_dict, min_node_size, min_num_nodes, length_display_text,
                               get_trace=False,
                               nx_title="Representation of pathway ontology information <br> Network of ontology–pathway associations",
                               nx_filename="pathway_ontology.html", heat_title="Pathway ontology heatmap",
                               heat_filename="pathway_ontology_heatmap.html", save_path='.'):
    G = nx.Graph()
    G.add_nodes_from(ontology_dict.keys())
    for key, val in ontology_dict.items():
        G.add_edges_from(([(key, t) for t in val if t in pathway_labels]))
    G.remove_nodes_from(list(nx.isolates(G)))

    fig = __generate_network_object(G=G, pos=None, min_node_size=min_node_size, get_trace=get_trace,
                                    disable_sizing=True, node_colorscale='YlGnBu', node_symbol='circle-dot',
                                    colorbar_x=1.02, xanchor='left', additional_dict=ontology_dict,
                                    additional_dict_symbol=False, additional_dict_tag="Ontology ID",
                                    min_num_nodes=min_num_nodes, title=nx_title, source=None,
                                    hover_name=PATHWAY_TAG)
    nx_filename = os.path.join(save_path, nx_filename)
    py.plot(fig, filename=nx_filename, auto_open=False)

    fig = __generate_heatmap_object(G=G, dict_object=ontology_dict, hover_name="Ontology ID",
                                    start_idx_color=0, length_display_text=length_display_text,
                                    title=heat_title)
    heat_filename = os.path.join(save_path, heat_filename)
    py.plot(fig, filename=heat_filename, auto_open=False)


def visualize_species_ontology_distr(df, clickable=False, filename='species_ontology_distr.html', save_path='.'):
    if not clickable:
        selector = alt.selection_single(empty='all', fields=['Pathway_ID'])

        base = alt.Chart(df).properties(
            width=400,
            height=400
        ).add_selection(selector)

        species_bars = base.mark_bar().encode(
            x=alt.X('mean(Species_Distribution):Q', stack='normalize'),
            y=alt.Y('Species_ID:N'),
            color=alt.Color('Pathway_ID'),
            tooltip=['Pathway_ID', 'Species_Distribution', 'Species_ID']
        ).interactive()
        species_bars.encoding.x.title = "Distribution"
        species_bars.encoding.y.title = "Species ID"

        ontology_bars = base.mark_bar().encode(
            x=alt.X('mean(Ontology_Distribution):Q', stack='normalize'),
            y=alt.Y('Ontology_ID:N'),
            color=alt.Color('Pathway_ID'),
            tooltip=['Pathway_ID', 'Ontology_Distribution', 'Ontology_ID']
        ).interactive().transform_filter(selector)
        ontology_bars.encoding.x.title = "Distribution"
        ontology_bars.encoding.y.title = "Ontology ID"
        (species_bars | ontology_bars).save(os.path.join(save_path, filename))
    else:
        species_bars = alt.Chart(df).transform_calculate(url=PATHWAY_URL + alt.datum.Pathway_ID).mark_bar().encode(
            x=alt.X('mean(Species_Distribution):Q', stack='normalize'),
            y=alt.Y('Species_ID:N'),
            href='url:N',
            color=alt.Color('Pathway_ID'),
            tooltip=['Pathway_ID', 'Species_Distribution', 'Species_ID', 'url:N']
        )
        species_text = alt.Chart(df).mark_text(dx=-25, dy=3, color='black').encode(
            x=alt.X('mean(Species_Distribution):Q', stack='normalize'),
            y=alt.Y('Species_ID:N'),
            detail='Pathway_ID:N',
            text=alt.Text('mean(Species_Distribution):Q', format='.4f')
        )
        species_bars.encoding.x.title = "Distribution"
        species_bars.encoding.y.title = "Species ID"
        ontology_bars = alt.Chart(df).transform_calculate(url=PATHWAY_URL + alt.datum.Pathway_ID).mark_bar().encode(
            x=alt.X('mean(Ontology_Distribution):Q', stack='normalize'),
            y=alt.Y('Ontology_ID:N'),
            href='url:N',
            color=alt.Color('Pathway_ID'),
            tooltip=['Pathway_ID', 'Ontology_Distribution', 'Ontology_ID', 'url:N']
        )
        ontology_text = alt.Chart(df).mark_text(dx=-25, dy=3, color='black').encode(
            x=alt.X('mean(Ontology_Distribution):Q', stack='normalize'),
            y=alt.Y('Ontology_ID:N'),
            detail='Pathway_ID:N',
            text=alt.Text('mean(Ontology_Distribution):Q', format='.4f')
        )
        ontology_bars.encoding.x.title = "Distribution"
        ontology_bars.encoding.y.title = "Ontology ID"
        (species_bars + species_text | ontology_bars + ontology_text).save(os.path.join(save_path, filename))


def visualize_distr(df, x_val, distr_val, get_trace, y_title, title, is_bar=True,
                    filename="species_ontologies_distr_interactive.html", save_path='.'):
    if is_bar:
        fig = __generate_bar_object(df=df, x_val=x_val, y_val='Pathway_ID', distr_val=distr_val,
                                    get_trace=get_trace, y_title=y_title, title=title)
    else:
        fig = __generate_pie_object(df=df, x_val=x_val, y_val='Pathway_ID', distr_val=distr_val,
                                    get_trace=get_trace, title=title)
    if get_trace:
        return fig
    filename = os.path.join(save_path, filename)
    py.plot(fig, filename=filename, auto_open=False)


def visualize_hin(hin, pathway_labels, ec_labels, mol_labels, min_node_size, min_num_nodes,
                  title="Representation of heterogeneous information network of MetaCyc", filename="hin.html",
                  save_path='.'):
    compose_labels = list(ec_labels)
    compose_labels.extend(mol_labels)
    compose_labels.extend(pathway_labels)
    H = nx.subgraph(G=hin, nbunch=compose_labels)
    alt_names = {'T': 'Pathway', 'C': 'Compound', 'E': 'EC'}
    hin_dict = dict()
    for node in H.nodes(data=True):
        tmp = node[1]['type']
        if alt_names[tmp] in hin_dict.keys():
            hin_dict[alt_names[tmp]] += [node[0]]
        else:
            hin_dict.update({alt_names[tmp]: [node[0]]})
    G = nx.Graph()
    G.add_nodes_from(hin_dict.keys())
    for key, val in hin_dict.items():
        G.add_edges_from(([(key, t) for t in val if t in compose_labels]))
        G.add_edges_from(([(s, t) for s in val if s in compose_labels for _, t in H.edges(s) if t in val]))
    G.add_edges_from(
        ([(alt_names[H.nodes(data=True)[s]['type']], alt_names[H.nodes(data=True)[t]['type']]) for s, t in H.edges()
          if H.nodes(data=True)[s]['type'] != H.nodes(data=True)[t]['type']]))
    fig = __generate_network_object(G=G, pos=None, min_node_size=min_node_size, get_trace=False,
                                    disable_sizing=True, node_colorscale='YlGnBu', node_symbol='circle-dot',
                                    colorbar_x=1.02, xanchor='left', additional_dict=hin_dict,
                                    additional_dict_symbol=False, additional_dict_tag="HIN ID",
                                    min_num_nodes=min_num_nodes, title=title, source=None, hover_name="ID")
    filename = os.path.join(save_path, filename)
    py.plot(fig, filename=filename, auto_open=False)


def visualize_embeddings(hin, P, perplexity, n_iter, min_node_size, min_num_nodes,
                         title="Representation of embeddings of pathways <br> Network of pathway–pathway associations",
                         filename="pathway_embedding.html", save_path='.'):
    pathway_idx = np.array([node[1]['pidx'] for node in hin.nodes(data=True) if node[1]['type'] == "T"])
    pathway_idx = pathway_idx[:10]
    pathway_labels = np.array([node[0] for node in hin.nodes(data=True)
                               if node[1]['type'] == "T" and node[1]['pidx'] in pathway_idx])
    tsne = manifold.TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, init="pca", random_state=12345)
    pos = tsne.fit_transform(P[pathway_idx, :])
    pos = dict(zip(pathway_labels, pos))
    G = nx.subgraph(G=hin, nbunch=pathway_labels)
    fig = __generate_network_object(G=G, pos=pos, min_node_size=min_node_size, get_trace=False,
                                    disable_sizing=True, node_colorscale='YlGnBu', node_symbol='circle-dot',
                                    colorbar_x=1.02, xanchor='left', additional_dict=None,
                                    additional_dict_symbol=False, additional_dict_tag=False,
                                    min_num_nodes=min_num_nodes, title=title,
                                    source=None, hover_name=PATHWAY_TAG)
    filename = os.path.join(save_path, filename)
    py.plot(fig, filename=filename, auto_open=False)


def __build_plots(hin, folder_path, pathway_info, pathway2ec, pathway2ec_idx, P, num_labels2plot,
                  min_node_size, min_num_nodes, length_display_text, perplexity, num_epochs, visualize_embeddings,
                  batch_idx, total_progress):
    desc = '\t\t--> Building plots {0:.4f}%...'.format(((batch_idx + 1) / total_progress * 100))
    print(desc, end="\r")

    for sidx, save_path in enumerate(folder_path):
        df_sample_pathways = pd.read_csv(os.path.join(save_path, 'pathway_report.csv'), sep='\t')
        pathway_labels = list(df_sample_pathways['PathwayFrameID'])

        # extract various pathway related information
        pathway_labels = np.array([node[0] for node in hin.nodes(data=True)
                                   if node[1]['type'] == "T" and node[0] in pathway_labels])
        pathway_idx = np.array([node[1]['pidx'] for node in hin.nodes(data=True) if node[1]['type'] == "T" and
                                node[0] in pathway_labels])
        if num_labels2plot >= pathway_labels.shape[0]:
            num_labels2plot = pathway_labels.shape[0]
        pathway_labels = np.random.choice(a=pathway_labels, size=num_labels2plot, replace=False)

        ec_idx = pathway2ec_idx[np.unique(np.where(pathway2ec[pathway_idx].toarray() != 0)[1])]
        ec_labels = np.array([node[0] for node in hin.nodes(data=True)
                              if node[1]['type'] == "E" and node[1]['pidx'] in ec_idx])

        mol_labels = [edge[1] for node in hin.nodes(data=True) if node[0] in pathway_labels for edge in
                      hin.edges(node[0])]
        mol_labels = np.unique([node[0] for node in hin.nodes(data=True)
                                if node[1]['type'] == "C" and node[0] in mol_labels])

        tmp = __extract_information(hin, pathway_info)
        onto_dict, ptwy_onto_dict, species_dict, ptwy_species_dict, tax_dict, ptwy_tax_dict = tmp

        # define dataframe
        merged_df, ontology_df, species_df = __get_dataframe(pathway_labels=pathway_labels,
                                                             species_dict=species_dict,
                                                             ontology_dict=onto_dict,
                                                             pathway_species_dict=ptwy_species_dict,
                                                             pathway_ontology_dict=ptwy_onto_dict)

        # visualize embeddings
        if visualize_embeddings:
            visualize_embeddings(hin=hin, P=P, perplexity=perplexity, n_iter=num_epochs, min_node_size=min_node_size,
                                 min_num_nodes=min_num_nodes, save_path=save_path)

        # visualize heterogeneous information network of MetaCyc
        visualize_hin(hin=hin, pathway_labels=pathway_labels, ec_labels=ec_labels, mol_labels=mol_labels,
                      min_node_size=min_node_size, min_num_nodes=min_num_nodes, save_path=save_path)

        # visualize pathway ontology
        visualize_pathway_ontology(pathway_labels=pathway_labels, ontology_dict=onto_dict, min_node_size=min_node_size,
                                   min_num_nodes=min_num_nodes, length_display_text=length_display_text,
                                   save_path=save_path)

        # visualize known species and their associated pathways
        visualize_species2pathways_heat(pathway_labels=pathway_labels, species_dict=species_dict,
                                        length_display_text=length_display_text, save_path=save_path)

        # visualize pathways and their associated ECs
        visualize_layer(hin=hin, labels=pathway_labels, min_node_size=min_node_size, min_num_nodes=min_num_nodes,
                        save_path=save_path)
        visualize_layer(hin=hin, labels=ec_labels, min_node_size=min_node_size, min_num_nodes=min_num_nodes,
                        title="EC-EC Interaction Network", hover_name=EC_TAG, filename="ec.html",
                        save_path=save_path)

        # visualize species and ontologies distributions
        ## composed distribution
        visualize_species_ontology_distr(df=merged_df, filename='species_ontologies_distr_interactive.html',
                                         save_path=save_path)
        visualize_species_ontology_distr(df=merged_df, clickable=True,
                                         filename='species_ontology_distr_interactive_url.html',
                                         save_path=save_path)

        ## get distribution of species by pathways as bar
        visualize_distr(df=merged_df, x_val='Species_ID', distr_val="Species_Distribution", get_trace=False,
                        y_title='Species', title="Distribution of species by pathways", is_bar=True,
                        filename="species_distr.html", save_path=save_path)
        ## get distribution of species by pathways as pie
        visualize_distr(df=merged_df, x_val='Species_ID', distr_val="Species_Distribution", get_trace=False,
                        y_title='Species', title="Distribution of species by pathways", is_bar=False,
                        filename="species_distr_pie.html", save_path=save_path)

        ## get distribution of ontologies by pathways as bar
        visualize_distr(df=merged_df, x_val='Ontology_ID', distr_val="Ontology_Distribution", get_trace=False,
                        y_title='Ontology', title="Distribution of ontologies by pathways", is_bar=True,
                        filename="ontology_distr.html", save_path=save_path)
        ## get distribution of ontologies by pathways as pie
        visualize_distr(df=merged_df, x_val='Ontology_ID', distr_val="Ontology_Distribution", get_trace=False,
                        y_title='Ontology', title="Distribution of ontologies by pathways", is_bar=False,
                        filename="ontology_distr_pie.html", save_path=save_path)

        ## get distribution of ontologies and species by pathways as pie
        trace_1 = visualize_distr(df=merged_df, x_val='Species_ID', distr_val="Species_Distribution", get_trace=True,
                                  title="Distribution of species by pathways", is_bar=False)
        trace_2 = visualize_distr(df=merged_df, x_val='Ontology_ID', distr_val="Ontology_Distribution", get_trace=True,
                                  title="Distribution of ontologies by pathways", is_bar=False)

        # make subplots of two figures
        fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'domain'}, {'type': 'domain'}]])
        fig.append_trace(trace_1, 1, 1)
        fig.append_trace(trace_2, 1, 2)

        # Set theme, margin, and annotation in layout
        layout = dict(width=1200, height=400, title=None, titlefont_size=16, showlegend=False,
                      hovermode='closest', margin=dict(b=20, l=100, r=100, t=40),
                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        fig.update_layout(layout)
        py.plot(fig, filename=os.path.join(save_path, "species_ontologies_distr.html"), auto_open=False)


def build_plots(hin, pathway_info, pathway2ec, pathway2ec_idx, P, num_labels2plot, min_node_size, min_num_nodes,
                length_display_text, perplexity=50, num_epochs=5000, visualize_embeddings=False, batch_size=30,
                num_jobs=1, rsfolder='Results', rspath='.'):
    main_folder_path = os.path.join(rspath, rsfolder)
    sample_folder_path = [os.path.join(main_folder_path, o) for o in os.listdir(main_folder_path)
                          if os.path.isdir((os.path.join(main_folder_path, o)))]
    num_samples = len(sample_folder_path)
    list_batches = np.arange(start=0, stop=num_samples, step=batch_size)
    parallel = Parallel(n_jobs=num_jobs, verbose=0)
    results = parallel(delayed(__build_plots)(hin, sample_folder_path[batch:batch + batch_size],
                                              pathway_info, pathway2ec, pathway2ec_idx,
                                              P, num_labels2plot, min_node_size,
                                              min_num_nodes, length_display_text,
                                              perplexity, num_epochs, visualize_embeddings,
                                              batch_idx, len(list_batches))
                       for batch_idx, batch in enumerate(list_batches))

    desc = '\t\t--> Building plots {0:.4f}%...'.format(100)
    print(desc)


###***************************          tSNE Embedding          ***************************###

def tsne_embedding(embeds, labels, labeled_points=None, num_components=2, perplexity=30, num_epochs=1000,
                   num_points_to_plot=500, num_points_to_label=100, file_name='embedding', save_path="."):
    try:
        def plot_with_labels(low_dim_embs, points2label_idx, labels, save_file):
            plt.figure(figsize=(18, 18))  # in inches
            for i in np.arange(low_dim_embs.shape[0]):
                x, y = low_dim_embs[i, :]
                plt.scatter(x, y)
                label = None
                if i in points2label_idx:
                    label = labels[np.where(points2label_idx == i)[0]][0]
                plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points',
                             ha='right', va='bottom')
            plt.axis('tight')
            plt.savefig(save_file)

        if len(labels) <= num_points_to_plot:
            print('\t>> More labels than embeddings. Setting the num_points_to_plot to the number of labels.')
            num_points_to_plot = len(labels)
        if num_points_to_plot <= num_points_to_label or labeled_points is not None:
            desc = ""
            desc += '\t>> Less points to label. Setting the num_points_to_label argument to the number of points.'
            num_points_to_label = num_points_to_plot
            if labeled_points is not None:
                if num_points_to_plot >= len(labeled_points):
                    desc += "\t>> Less points to plot w.r.t the number of labeled points. "
                    desc += "Setting the num_points_to_plot argument to the provided labeled points"
                    num_points_to_plot = len(labeled_points)

        idx_labels = labeled_points
        if labeled_points is None:
            idx_labels = np.random.choice(a=len(labels), size=num_points_to_plot, replace=False)

        points2label_idx = np.random.choice(a=idx_labels, size=num_points_to_label, replace=False)
        labels = labels[points2label_idx]
        tsne = manifold.TSNE(n_components=num_components, perplexity=perplexity, n_iter=num_epochs,
                             init="pca", random_state=12345)
        low_dim_embs = tsne.fit_transform(embeds[idx_labels, :])
        file = os.path.join(save_path, file_name + '_tsne')
        print('\t\t## Storing {0:s} into the file: {1:s}'.format(file_name, file))
        plot_with_labels(low_dim_embs, points2label_idx=points2label_idx, labels=labels, save_file=file)
    except ImportError as ex:
        print('Please install sklearn, matplotlib, TSNE, and scipy to show embeddings.')
        print(ex)


###***************************         Community Matrix         ***************************###

def __get_comm_matrix(G):
    comms_clustering = community.best_partition(G)
    n_nodes = nx.number_of_nodes(G)
    labels = sorted(set([label[1] for label in comms_clustering.items()]))
    M = dok_matrix((n_nodes, len(labels)), dtype=np.bool)
    for node_idx, label in comms_clustering.items():
        comm_idx = labels.index(label)
        M[node_idx, comm_idx] = True
    return M.tocsr(), labels


def __assign_array2lists(X):
    by_attribute_val = defaultdict(list)
    for col_idx in np.arange(X.shape[1]):
        by_attribute_val[col_idx].append(X[:, col_idx].nonzero()[0])
    return by_attribute_val.values()


def __draw_adjacency_matrix(G, node_order=None, partitions=None, colors=[], file_name='', save_path='.'):
    """
    - G is a networkx graph
    - node_order (optional) is a list of nodes, where each node in G
          appears exactly once
    - partitions is a list of node lists, where each node in G appears
          in exactly one node list
    - colors is a list of strings indicating what color each
          partition should be
    If partitions is specified, the same number of colors needs to be
    specified.
    """
    if partitions is None:
        partitions = []
    adjacency_matrix = nx.to_numpy_matrix(G, dtype=np.bool, nodelist=node_order)

    # Plot adjacency matrix in toned-down black and white
    fig = pyplot.figure(figsize=(5, 5))  # in inches
    pyplot.imshow(adjacency_matrix,
                  cmap="Greys",
                  interpolation="none")

    # The rest is just if you have sorted nodes by a partition and want to
    # highlight the module boundaries
    assert len(partitions) == len(colors)
    ax = pyplot.gca()
    for partition, color in zip(partitions, colors):
        current_idx = 0
        for module in partition:
            ax.add_patch(patches.Rectangle((current_idx, current_idx),
                                           len(module),  # Width
                                           len(module),  # Height
                                           facecolor="none",
                                           edgecolor=color,
                                           linewidth="1"))
            current_idx += len(module)
    file_name = file_name + '_adj.eps'
    file = os.path.join(save_path, file_name)
    print('\t\t## Storing figure to: {0:s}'.format(file_name))
    fig.savefig(file)


def build_community(Adj, file_name, save_path='.'):
    np.fill_diagonal(Adj, 0)
    G = nx.from_numpy_matrix(Adj)
    comm_matrix, labels = __get_comm_matrix(G=G)
    nodes_labels_ordered = [node_id for comm_idx, comm_id in enumerate(labels)
                            for node_id in comm_matrix[:, comm_idx].nonzero()[0]]
    labels_list = __assign_array2lists(X=comm_matrix)
    labels_list = [node for comm in labels_list for node in comm]
    __draw_adjacency_matrix(G=G, node_order=nodes_labels_ordered, partitions=[labels_list], colors=["blue"],
                            file_name=file_name, save_path=save_path)


###***************************               Main               ***************************###

if __name__ == '__main__':
    os.system('clear')
    DIRECTORY_PATH = os.getcwd()
    REPO_PATH = DIRECTORY_PATH.split('/')
    REPO_PATH = '/'.join(REPO_PATH[:-3])
    dspath = os.path.join(REPO_PATH, 'dataset')
    rspath = os.path.join(REPO_PATH, 'result')
    ospath = os.path.join(REPO_PATH, 'objectset')
    # TODO: uncomment the following
    # P = load_data(comp_file_name="P_triNMF.pkl", load_path=dspath, tag='P_triNMF')
    # hin = load_data(comp_file_name="hin_uec_r_ex_77.pkl", load_path=ospath, tag='heterogeneous information network')
    # labels = np.array([node[0] for node in hin.nodes(data=True) if node[1]['type'] == "T"])
    # tsne_embedding(embeds=P, labels=labels, labeled_points=None, num_components=2, perplexity=50,
    #                num_epochs=5000, num_points_to_plot=P.shape[0], num_points_to_label=100,
    #                comp_file_name="embedding", save_path=rspath)
