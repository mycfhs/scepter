# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
from __future__ import annotations

import os.path
import time

import gradio as gr
import imagehash
from PIL import Image
from scepter.modules.utils.directory import get_md5
from scepter.modules.utils.file_system import FS
from scepter.studio.preprocess.caption_editor_ui.component_names import \
    DatasetGalleryUIName
from scepter.studio.preprocess.caption_editor_ui.create_dataset_ui import \
    CreateDatasetUI
from scepter.studio.preprocess.processors.processor_manager import \
    ProcessorsManager
from scepter.studio.utils.uibase import UIBase


class DatasetGalleryUI(UIBase):
    def __init__(self, cfg, is_debug=False, language='en', create_ins=None):
        self.selected_path = ''
        self.selected_index = -1
        self.selected_index_prev = -1

        self.processors_manager = ProcessorsManager(cfg.PROCESSORS,
                                                    language=language)

        self.component_names = DatasetGalleryUIName(language)
        if create_ins is not None:
            self.default_dataset = create_ins.default_dataset
            current_info = self.default_dataset.current_record
            self.default_ori_caption = current_info.get('caption', '')
            self.default_edit_caption = current_info.get('edit_caption', '')
            self.default_image_width = current_info.get('width', -1)
            self.default_image_height = current_info.get('height', -1)
            self.default_image_format = os.path.splitext(
                current_info.get('relative_path', ''))[-1]

            self.default_edit_image_width = current_info.get(
                'edit_width', current_info.get('width', -1))
            self.default_edit_image_height = current_info.get(
                'edit_height', current_info.get('height', -1))
            self.default_edit_image_format = os.path.splitext(
                current_info.get('edit_relative_path',
                                 current_info.get('relative_path', '')))[-1]

            self.default_select_index = self.default_dataset.cursor
            self.default_info = f'{self.default_dataset.cursor + 1}/{len(self.default_dataset)}'
            image_list = [
                os.path.join(self.default_dataset.meta['local_work_dir'],
                             v['relative_path'])
                for v in self.default_dataset.data
            ]
            self.default_image_list = image_list
            self.default_dataset_type = create_ins.default_dataset_type
        else:
            self.default_dataset = None
            self.default_ori_caption = ''
            self.default_edit_caption = ''
            self.default_image_width = -1
            self.default_image_height = -1
            self.default_image_format = -1
            self.default_edit_image_width = -1
            self.default_edit_image_height = -1
            self.default_edit_image_format = -1
            self.default_select_index = -1
            self.default_image_list = []
            self.default_info = ''
            self.default_dataset_type = 'scepter_txt2img'

    def create_ui(self):
        with gr.Row(variant='panel',
                    visible=self.default_dataset is not None,
                    equal_height=True) as gallery_panel:
            with gr.Row(visible=False):
                self.gallery_state = gr.Text(label='gallery_state',
                                             value='',
                                             visible=False)
                self.mode_state = gr.Text(label='mode_state',
                                          value='view',
                                          visible=False)
                self.src_dataset_state = gr.Checkbox(label='src_state',
                                                     value=False,
                                                     visible=False)
            with gr.Column(variant='panel', min_width=0,
                           visible=False) as self.edit_panel:

                with gr.Row():
                    self.edit_image_info = gr.Markdown(
                        self.component_names.edit_dataset.format(
                            self.default_edit_image_height,
                            self.default_edit_image_width,
                            self.default_edit_image_format))
                with gr.Row():
                    self.edit_caption = gr.Textbox(
                        label=self.component_names.edit_caption,
                        placeholder='',
                        value=self.default_edit_caption,
                        lines=4,
                        autoscroll=False,
                        interactive=True,
                        visible=False)
                with gr.Row(equal_height=True):
                    with gr.Column(scale=1, min_width=0,
                                   visible=False) as self.edit_src_panel:
                        self.edit_src_gl_dataset_images = gr.Gallery(
                            label=self.component_names.edit_dataset_src_images,
                            elem_id='dataset_tag_editor_dataset_src_gallery',
                            value=self.default_image_list,
                            selected_index=self.default_select_index,
                            columns=4,
                            visible=False,
                            interactive=False)
                    with gr.Column(scale=1, min_width=0):
                        self.edit_gl_dataset_images = gr.Gallery(
                            label=self.component_names.edit_dataset_images,
                            elem_id='dataset_tag_editor_dataset_gallery',
                            value=self.default_image_list,
                            selected_index=self.default_select_index,
                            object_fit='fill',
                            preview=True,
                            columns=4,
                            visible=False,
                            interactive=False)
                    # with gr.Column(ariant='panel', scale=2, min_width=0):

            with gr.Column(variant='panel', min_width=0):
                with gr.Row():
                    self.image_info = gr.Markdown(
                        self.component_names.ori_dataset.format(
                            self.default_image_height,
                            self.default_image_width,
                            self.default_image_format))
                with gr.Row():
                    self.ori_caption = gr.Textbox(
                        label=self.component_names.ori_caption,
                        placeholder='',
                        value=self.default_ori_caption,
                        lines=4,
                        autoscroll=False,
                        interactive=False)
                with gr.Row(equal_height=True):
                    with gr.Column(scale=1, min_width=0,
                                   visible=False) as self.src_panel:
                        self.src_gl_dataset_images = gr.Gallery(
                            label=self.component_names.dataset_src_images,
                            elem_id='dataset_tag_editor_dataset_src_gallery',
                            value=self.default_image_list,
                            selected_index=self.default_select_index,
                            columns=4,
                            visible=False,
                            interactive=False)
                    with gr.Column(scale=1, min_width=0):
                        with gr.Row():
                            self.gl_dataset_images = gr.Gallery(
                                label=self.component_names.dataset_images,
                                elem_id='dataset_tag_editor_dataset_gallery',
                                value=self.default_image_list,
                                selected_index=self.default_select_index,
                                columns=4,
                                preview=True,
                                allow_preview=True,
                                object_fit='fill')

        with gr.Row(visible=False) as self.edit_setting_panel:
            self.sys_log = gr.Markdown(
                self.component_names.system_log.format(''))
        with (gr.Row()):
            with gr.Column(variant='panel',
                           visible=False,
                           scale=1,
                           min_width=0) as self.edit_confirm_panel:
                with gr.Group():
                    with gr.Row():
                        gr.Markdown(self.component_names.confirm_direction)
                    with gr.Row():
                        self.range_mode = gr.Dropdown(
                            label=self.component_names.set_range_name,
                            choices=self.component_names.range_mode_name,
                            value=self.component_names.range_mode_name[0])
                    with gr.Row():
                        self.data_range = gr.Dropdown(
                            show_label=True,
                            label=self.component_names.
                            samples_range_placeholder,
                            choices=[],
                            allow_custom_value=True,
                            multiselect=True,
                            visible=False,
                            interactive=True)
                    with gr.Row():
                        with gr.Column(scale=1, min_width=0):
                            self.btn_confirm_edit = gr.Button(
                                value=self.component_names.btn_confirm_edit)
                        with gr.Column(scale=1, min_width=0):
                            self.btn_cancel_edit = gr.Button(
                                value=self.component_names.btn_cancel_edit)
                        with gr.Column(scale=1, min_width=0):
                            self.btn_reset_edit = gr.Button(
                                value=self.component_names.btn_reset_edit)
                    with gr.Row():
                        self.preprocess_checkbox = gr.CheckboxGroup(
                            show_label=False,
                            choices=self.component_names.preprocess_choices,
                            value=None)
            with gr.Column(variant='panel',
                           visible=False,
                           scale=2,
                           min_width=0) as self.upload_panel:
                with gr.Row():
                    self.upload_src_image_info = gr.Markdown(value='',
                                                             visible=False)
                with gr.Row():
                    self.upload_image_info = gr.Markdown(value='')

                with gr.Row():
                    self.upload_caption = gr.Textbox(
                        label=self.component_names.image_caption,
                        placeholder='',
                        value='',
                        lines=5)
                with gr.Row():
                    with gr.Column(
                            variant='panel',
                            scale=1,
                            min_width=0,
                            visible=self.default_dataset_type ==
                            'scepter_img2img') as self.upload_src_image_panel:
                        self.upload_src_image = gr.Image(
                            label=self.component_names.upload_src_image,
                            type='pil')
                    with gr.Column(variant='panel', scale=1, min_width=0):
                        self.upload_image = gr.Image(
                            label=self.component_names.upload_image,
                            sources=['upload'],
                            type='pil')
                with gr.Row():
                    with gr.Column(min_width=0):
                        self.upload_button = gr.Button(
                            value=self.component_names.upload_image_btn)
                    with gr.Column(min_width=0):
                        self.cancel_button = gr.Button(
                            value=self.component_names.cancel_upload_btn)
            with gr.Column(variant='panel',
                           visible=False,
                           scale=2,
                           min_width=0) as self.preprocess_panel:
                with gr.Row(
                        variant='panel',
                        visible=False,
                ) as self.image_preprocess_panel:
                    with gr.Group():
                        with gr.Column(variant='panel', min_width=0):
                            with gr.Row():
                                self.image_preprocess_method = gr.Dropdown(
                                    label=self.component_names.
                                    image_processor_type,
                                    choices=self.processors_manager.
                                    get_choices('image'),
                                    value=self.processors_manager.get_default(
                                        'image'),
                                    interactive=True)
                            image_processor_ins = self.processors_manager.get_processor(
                                'image',
                                self.processors_manager.get_default('image'))
                            with gr.Row():
                                default_height_ratio = image_processor_ins.system_para.get(
                                    'HEIGHT_RATIO', {})
                                self.height_ratio = gr.Slider(
                                    label=self.component_names.height_ratio,
                                    minimum=default_height_ratio.get('MIN', 1),
                                    maximum=default_height_ratio.get(
                                        'MAX', 10),
                                    step=default_height_ratio.get('STEP', 1),
                                    value=default_height_ratio.get('VALUE', 1),
                                    visible=len(default_height_ratio) > 0,
                                    interactive=True)
                            with gr.Row():
                                default_width_ratio = image_processor_ins.system_para.get(
                                    'WIDTH_RATIO', {})
                                self.width_ratio = gr.Slider(
                                    label=self.component_names.width_ratio,
                                    minimum=default_width_ratio.get('MIN', 1),
                                    maximum=default_width_ratio.get('MAX', 10),
                                    step=default_width_ratio.get('STEP', 1),
                                    value=default_width_ratio.get('VALUE', 1),
                                    visible=len(default_width_ratio) > 0,
                                    interactive=True)
                            with gr.Row():
                                self.image_preprocess_btn = gr.Button(
                                    value=self.component_names.
                                    image_preprocess_btn)
                with gr.Row(variant='panel',
                            visible=False) as self.caption_preprocess_panel:
                    with gr.Group():
                        with gr.Column(variant='panel', min_width=0):
                            with gr.Row():
                                self.caption_preprocess_method = gr.Dropdown(
                                    label=self.component_names.
                                    caption_processor_type,
                                    choices=self.processors_manager.
                                    get_choices('caption'),
                                    value=self.processors_manager.get_default(
                                        'caption'),
                                    interactive=True)
                            with gr.Row():
                                with gr.Column(scale=1, min_width=0):
                                    self.caption_use_device = gr.Text(
                                        label=self.component_names.used_device,
                                        value=self.processors_manager.
                                        get_default_device('caption'),
                                        interactive=False)
                                with gr.Column(scale=1, min_width=0):
                                    self.caption_use_memory = gr.Text(
                                        label=self.component_names.used_memory,
                                        value=self.processors_manager.
                                        get_default_memory('caption'),
                                        interactive=False)
                                default_processor_method = self.processors_manager.get_default(
                                    'caption')
                                default_processor_ins = self.processors_manager.get_processor(
                                    'caption', default_processor_method)
                                with gr.Column(scale=1, min_width=0):
                                    self.caption_language = gr.Dropdown(
                                        label=self.component_names.
                                        caption_language,
                                        choices=default_processor_ins.
                                        get_language_choice,
                                        value=default_processor_ins.
                                        get_language_default)
                                with gr.Column(scale=1, min_width=0):
                                    self.caption_update_mode = gr.Dropdown(
                                        label=self.component_names.
                                        caption_update_mode,
                                        choices=self.component_names.
                                        caption_update_choices,
                                        value=self.component_names.
                                        caption_update_choices[0]
                                        if len(self.component_names.
                                               caption_update_choices) > 0 else
                                        None)
                            with gr.Accordion(
                                    label=self.component_names.advance_setting,
                                    open=False):
                                with gr.Row():
                                    self.sys_prompt = gr.Text(
                                        label=self.component_names.
                                        system_prompt,
                                        interactive=True,
                                        value=default_processor_ins.
                                        get_para_by_language(
                                            default_processor_ins.
                                            get_language_default).get(
                                                'PROMPT', ''),
                                        lines=2,
                                        visible='PROMPT'
                                        in default_processor_ins.
                                        get_para_by_language(
                                            default_processor_ins.
                                            get_language_default))
                                with gr.Row():
                                    default_max_new_tokens = default_processor_ins.get_para_by_language(
                                        default_processor_ins.
                                        get_language_default).get(
                                            'MAX_NEW_TOKENS', {})
                                    self.max_new_tokens = gr.Slider(
                                        label=self.component_names.
                                        max_new_tokens,
                                        minimum=default_max_new_tokens.get(
                                            'MIN', 0),
                                        maximum=default_max_new_tokens.get(
                                            'MAX', 1.0),
                                        step=default_max_new_tokens.get(
                                            'STEP', 0.1),
                                        value=default_max_new_tokens.get(
                                            'VALUE', 0.1),
                                        visible=len(default_max_new_tokens) >
                                        0,
                                        interactive=True)
                                with gr.Row():
                                    default_min_new_tokens = default_processor_ins.get_para_by_language(
                                        default_processor_ins.
                                        get_language_default).get(
                                            'MIN_NEW_TOKENS', {})

                                    self.min_new_tokens = gr.Slider(
                                        label=self.component_names.
                                        min_new_tokens,
                                        minimum=default_min_new_tokens.get(
                                            'MIN', 0),
                                        maximum=default_min_new_tokens.get(
                                            'MAX', 1.0),
                                        step=default_min_new_tokens.get(
                                            'STEP', 0.1),
                                        value=default_min_new_tokens.get(
                                            'VALUE', 0.1),
                                        visible=len(default_min_new_tokens) >
                                        0,
                                        interactive=True)
                                with gr.Row():
                                    default_num_beams = default_processor_ins.get_para_by_language(
                                        default_processor_ins.
                                        get_language_default).get(
                                            'NUM_BEAMS', {})

                                    self.num_beams = gr.Slider(
                                        label=self.component_names.num_beams,
                                        minimum=default_num_beams.get(
                                            'MIN', 0),
                                        maximum=default_num_beams.get(
                                            'MAX', 1.0),
                                        step=default_num_beams.get(
                                            'STEP', 0.1),
                                        value=default_num_beams.get(
                                            'VALUE', 0.1),
                                        visible=len(default_num_beams) > 0,
                                        interactive=True)
                                with gr.Row():
                                    default_repetition_penalty = default_processor_ins.get_para_by_language(
                                        default_processor_ins.
                                        get_language_default).get(
                                            'REPETITION_PENALTY', {})

                                    self.repetition_penalty = gr.Slider(
                                        label=self.component_names.
                                        repetition_penalty,
                                        minimum=default_repetition_penalty.get(
                                            'MIN', 0),
                                        maximum=default_repetition_penalty.get(
                                            'MAX', 1.0),
                                        step=default_repetition_penalty.get(
                                            'STEP', 0.1),
                                        value=default_repetition_penalty.get(
                                            'VALUE', 0.1),
                                        visible=len(default_repetition_penalty)
                                        > 0,
                                        interactive=True)
                                with gr.Row():
                                    default_temperature = default_processor_ins.get_para_by_language(
                                        default_processor_ins.
                                        get_language_default).get(
                                            'TEMPERATURE', {})

                                    self.temperature = gr.Slider(
                                        label=self.component_names.temperature,
                                        minimum=default_temperature.get(
                                            'MIN', 0),
                                        maximum=default_temperature.get(
                                            'MAX', 1.0),
                                        step=default_temperature.get(
                                            'STEP', 0.1),
                                        value=default_temperature.get(
                                            'VALUE', 0.1),
                                        visible=len(default_temperature) > 0,
                                        interactive=True)

                            with gr.Row():

                                with gr.Column(scale=1, min_width=0):
                                    self.caption_preprocess_btn = gr.Button(
                                        value=self.component_names.
                                        caption_preprocess_btn)

        with gr.Row(variant='panel', equal_height=True):
            with gr.Column(scale=1, min_width=0):
                self.info = gr.Text(value=self.default_info,
                                    label=None,
                                    show_label=False,
                                    container=False,
                                    interactive=False)
            with gr.Column(scale=1, min_width=0):
                self.modify_button = gr.Button(
                    value=self.component_names.btn_modify)
            with gr.Column(scale=1, min_width=0):
                self.add_button = gr.Button(value=self.component_names.btn_add)
            with gr.Column(scale=1, min_width=0):
                self.delete_button = gr.Button(
                    value=self.component_names.btn_delete)

        self.gallery_panel = gallery_panel

    def set_callbacks(self, create_dataset: CreateDatasetUI, manager):
        def range_state_trans(range_mode):
            reverse_mode = {
                v: id
                for id, v in enumerate(self.component_names.range_mode_name)
            }
            hit_range_mode = reverse_mode[range_mode]
            return hit_range_mode

        # Used to change the target gallery state( is not used now)
        def change_gallery(dataset_type, dataset_name):
            dataset_type = create_dataset.get_trans_dataset_type(dataset_type)
            dataset_ins = create_dataset.dataset_dict.get(
                dataset_type, {}).get(dataset_name, None)
            if dataset_ins is None:
                return gr.Gallery(), gr.Gallery(), gr.Column(), gr.Text()
            cursor = dataset_ins.cursor
            image_list = [
                os.path.join(dataset_ins.meta['local_work_dir'],
                             v['relative_path']) for v in dataset_ins.data
            ]
            if dataset_type == 'scepter_img2img':
                src_image_list = [
                    os.path.join(dataset_ins.meta['local_work_dir'],
                                 v['src_relative_path'])
                    for v in dataset_ins.data
                ]
                if cursor >= 0:
                    ret_src_gallery = gr.Gallery(label=dataset_name,
                                                 value=src_image_list,
                                                 selected_index=cursor,
                                                 visible=True)
                else:
                    ret_src_gallery = gr.Gallery(label=dataset_name,
                                                 value=src_image_list,
                                                 selected_index=None,
                                                 visible=True)
                ret_src_panel = gr.Column(visible=True)
            else:
                ret_src_gallery = gr.Gallery(visible=False)
                ret_src_panel = gr.Column(visible=False)

            if cursor >= 0:
                return (gr.Gallery(label=dataset_name,
                                   value=image_list,
                                   selected_index=cursor), ret_src_gallery,
                        ret_src_panel, gr.Text(value='view'))
            else:
                return (gr.Gallery(label=dataset_name,
                                   value=image_list,
                                   selected_index=None), ret_src_gallery,
                        ret_src_panel, gr.Text(value='view'))

        self.gallery_state.change(
            change_gallery,
            inputs=[create_dataset.dataset_type, create_dataset.dataset_name],
            outputs=[
                self.gl_dataset_images, self.src_gl_dataset_images,
                self.src_panel, self.mode_state
            ],
            queue=False)

        # select one target image return new select index for source image and target image
        def select_image(dataset_type, dataset_name, evt: gr.SelectData):
            dataset_type = create_dataset.get_trans_dataset_type(dataset_type)
            dataset_ins = create_dataset.dataset_dict.get(
                dataset_type, {}).get(dataset_name, None)
            if dataset_ins is None:
                return gr.Gallery(), gr.Gallery(), gr.Textbox(), '', gr.Text()
            dataset_ins.set_cursor(evt.index)
            current_info = dataset_ins.current_record
            all_number = len(dataset_ins)
            image_list = [
                os.path.join(dataset_ins.meta['local_work_dir'],
                             v['relative_path']) for v in dataset_ins.data
            ]
            ret_image_gl = gr.Gallery(
                value=image_list,
                selected_index=dataset_ins.cursor) if dataset_ins.cursor >= 0 \
                else gr.Gallery(value=[], selected_index=None)
            if dataset_type == 'scepter_img2img':
                src_image_list = [
                    os.path.join(dataset_ins.meta['local_work_dir'],
                                 v['src_relative_path'])
                    for v in dataset_ins.data
                ]
                ret_src_image_gl = gr.Gallery(value=src_image_list, selected_index=dataset_ins.cursor, visible=True) \
                    if dataset_ins.cursor >= 0 else gr.Gallery(value=[], selected_index=None)
            else:
                ret_src_image_gl = gr.Gallery(visible=False)

            ret_caption = gr.Textbox(value=current_info.get('caption', ''))
            ret_info = gr.Text(value=f'{dataset_ins.cursor+1}/{all_number}')

            ret_image_height = current_info.get('height', -1)
            ret_image_width = current_info.get('width', -1)
            ret_image_format = os.path.splitext(
                current_info.get('relative_path', ''))[-1]

            image_info = self.component_names.ori_dataset.format(
                ret_image_height, ret_image_width, ret_image_format)

            return (ret_image_gl, ret_src_image_gl, ret_caption, image_info,
                    ret_info)

        self.gl_dataset_images.select(
            select_image,
            inputs=[create_dataset.dataset_type, create_dataset.dataset_name],
            outputs=[
                self.gl_dataset_images, self.src_gl_dataset_images,
                self.ori_caption, self.image_info, self.info
            ],
            queue=False)

        # select one target image return new select index for source image and target image in edit mode
        def edit_select_image(dataset_type, dataset_name, mode_state,
                              evt: gr.SelectData):
            dataset_type = create_dataset.get_trans_dataset_type(dataset_type)
            dataset_ins = create_dataset.dataset_dict.get(
                dataset_type, {}).get(dataset_name, None)
            if dataset_ins is None:
                return gr.Gallery(), gr.Gallery(), gr.Gallery(), gr.Gallery(
                ), gr.Textbox(), '', gr.Text()

            cursor = dataset_ins.cursor_from_edit_index(evt.index)
            dataset_ins.set_cursor(cursor)
            current_info = dataset_ins.current_record
            all_number = len(dataset_ins)

            ret_image_gl = gr.Gallery(selected_index=dataset_ins.cursor) if dataset_ins.cursor >= 0 \
                else gr.Gallery(value=[], selected_index=None)
            dataset_ins.edit_cursor = evt.index
            ret_edit_image_gl = gr.Gallery(
                selected_index=dataset_ins.edit_cursor)

            if dataset_type == 'scepter_img2img':
                ret_src_image_gl = gr.Gallery(selected_index=dataset_ins.cursor, visible=True) \
                    if dataset_ins.cursor >= 0 \
                    else gr.Gallery(value=[], selected_index=None, visible=True)
                ret_src_edit_image_gl = gr.Gallery(selected_index=dataset_ins.edit_cursor, visible=True) \
                    if dataset_ins.cursor >= 0 \
                    else gr.Gallery(value=[], selected_index=None, visible=True)
            else:
                ret_src_image_gl = gr.Gallery(visible=False)
                ret_src_edit_image_gl = gr.Gallery(visible=False)

            ret_edit_caption = gr.Textbox(value=current_info.get(
                'edit_caption', ''),
                                          visible=True)

            ret_edit_image_width = current_info.get(
                'edit_width', current_info.get('width', -1))
            ret_edit_image_height = current_info.get(
                'edit_height', current_info.get('height', -1))
            ret_edit_image_format = os.path.splitext(
                current_info.get('edit_relative_path',
                                 current_info.get('relative_path', '')))[-1]

            edit_image_info = self.component_names.edit_dataset.format(
                ret_edit_image_height, ret_edit_image_width,
                ret_edit_image_format)

            ret_info = gr.Text(value=f'{dataset_ins.cursor+1}/{all_number}')
            return (ret_image_gl, ret_src_image_gl, ret_edit_image_gl,
                    ret_src_edit_image_gl, ret_edit_caption, edit_image_info,
                    ret_info)

        self.edit_gl_dataset_images.select(
            edit_select_image,
            inputs=[
                create_dataset.dataset_type, create_dataset.dataset_name,
                self.mode_state
            ],
            outputs=[
                self.gl_dataset_images, self.src_gl_dataset_images,
                self.edit_gl_dataset_images, self.edit_src_gl_dataset_images,
                self.edit_caption, self.edit_image_info, self.info
            ],
            queue=False)

        def src_select_image(dataset_type, dataset_name, evt: gr.SelectData):
            dataset_type = create_dataset.get_trans_dataset_type(dataset_type)
            dataset_ins = create_dataset.dataset_dict.get(
                dataset_type, {}).get(dataset_name, None)
            if dataset_ins is None:
                return gr.Gallery(), gr.Gallery(), gr.Textbox(), '', gr.Text()
            dataset_ins.set_cursor(evt.index)
            ret_image_gl = gr.Gallery(selected_index=dataset_ins.cursor) if dataset_ins.cursor >= 0 \
                else gr.Gallery(value=[], selected_index=None)

            return ret_image_gl

        self.src_gl_dataset_images.select(
            src_select_image,
            inputs=[create_dataset.dataset_type, create_dataset.dataset_name],
            outputs=[self.gl_dataset_images],
            queue=False)

        def edit_src_select_image(dataset_type, dataset_name, mode_state,
                                  evt: gr.SelectData):
            dataset_type = create_dataset.get_trans_dataset_type(dataset_type)
            dataset_ins = create_dataset.dataset_dict.get(
                dataset_type, {}).get(dataset_name, None)
            if dataset_ins is None:
                return gr.Gallery()
            cursor = dataset_ins.cursor_from_edit_index(evt.index)
            dataset_ins.set_cursor(cursor)
            dataset_ins.edit_cursor = evt.index
            ret_edit_image_gl = gr.Gallery(
                selected_index=dataset_ins.edit_cursor)
            return ret_edit_image_gl

        self.edit_src_gl_dataset_images.select(
            edit_src_select_image,
            inputs=[
                create_dataset.dataset_type, create_dataset.dataset_name,
                self.mode_state
            ],
            outputs=[self.edit_gl_dataset_images],
            queue=False)

        # target image gallery changed
        def change_gallery(dataset_type, dataset_name):
            dataset_type = create_dataset.get_trans_dataset_type(dataset_type)
            dataset_ins = create_dataset.dataset_dict.get(
                dataset_type, {}).get(dataset_name, None)
            if dataset_ins is None:
                return gr.Textbox(), '', gr.Textbox(), '', gr.Text()
            current_info = dataset_ins.current_record
            ret_image_height = current_info.get('height', -1)
            ret_image_width = current_info.get('width', -1)
            ret_image_format = os.path.splitext(
                current_info.get('relative_path', ''))[-1]

            image_info = self.component_names.ori_dataset.format(
                ret_image_height, ret_image_width, ret_image_format)

            ret_edit_image_width = current_info.get(
                'edit_width', current_info.get('width', -1))
            ret_edit_image_height = current_info.get(
                'edit_height', current_info.get('height', -1))
            ret_edit_image_format = os.path.splitext(
                current_info.get('edit_relative_path',
                                 current_info.get('relative_path', '')))[-1]

            edit_image_info = self.component_names.edit_dataset.format(
                ret_edit_image_height, ret_edit_image_width,
                ret_edit_image_format)

            all_number = len(dataset_ins)
            return (gr.Textbox(value=current_info.get('caption', '')),
                    image_info,
                    gr.Textbox(value=current_info.get('edit_caption', '')),
                    edit_image_info,
                    gr.Text(value=f'{dataset_ins.cursor+1}/{all_number}'))

        self.gl_dataset_images.change(
            change_gallery,
            inputs=[create_dataset.dataset_type, create_dataset.dataset_name],
            outputs=[
                self.ori_caption, self.image_info, self.edit_caption,
                self.edit_image_info, self.info
            ],
            queue=False)

        def edit_mode():
            return gr.Text(value='edit')

        self.modify_button.click(edit_mode,
                                 inputs=[],
                                 outputs=[self.mode_state],
                                 queue=False)

        def view_mode():
            return gr.Text(value='view')

        self.btn_cancel_edit.click(view_mode,
                                   inputs=[],
                                   outputs=[self.mode_state],
                                   queue=False)

        # reset edit information to clean the status of editing
        def reset_edit(dataset_type, dataset_name):
            dataset_type = create_dataset.get_trans_dataset_type(dataset_type)
            dataset_ins = create_dataset.dataset_dict.get(
                dataset_type, {}).get(dataset_name, None)
            if dataset_ins is None:
                return gr.Gallery(), gr.Text(), ''
            for idx, one_data in enumerate(dataset_ins.edit_samples):
                current_cursor = dataset_ins.cursor_from_edit_index(idx)
                dataset_ins.edit_samples[idx][
                    'edit_relative_path'] = dataset_ins.data[current_cursor][
                        'relative_path']
                dataset_ins.edit_samples[idx][
                    'edit_image_path'] = dataset_ins.data[current_cursor][
                        'image_path']
                dataset_ins.edit_samples[idx][
                    'edit_caption'] = dataset_ins.data[current_cursor][
                        'caption']
                dataset_ins.edit_samples[idx]['edit_width'] = dataset_ins.data[
                    current_cursor]['width']
                dataset_ins.edit_samples[idx][
                    'edit_height'] = dataset_ins.data[current_cursor]['height']
                if dataset_type == 'scepter_img2img':
                    dataset_ins.edit_samples[idx][
                        'edit_src_relative_path'] = dataset_ins.data[
                            current_cursor]['src_relative_path']
                    dataset_ins.edit_samples[idx][
                        'edit_src_image_path'] = dataset_ins.data[
                            current_cursor]['src_image_path']
                    dataset_ins.edit_samples[idx][
                        'edit_src_width'] = dataset_ins.data[current_cursor][
                            'src_width']
                    dataset_ins.edit_samples[idx][
                        'edit_src_height'] = dataset_ins.data[current_cursor][
                            'src_height']

            image_list = [
                os.path.join(dataset_ins.meta['local_work_dir'],
                             v.get('edit_relative_path', v['relative_path']))
                for v in dataset_ins.edit_samples
            ]
            current_info = dataset_ins.current_record

            ret_edit_image_width = current_info.get(
                'edit_width', current_info.get('width', -1))
            ret_edit_image_height = current_info.get(
                'edit_height', current_info.get('height', -1))
            ret_edit_image_format = os.path.splitext(
                current_info.get('edit_relative_path',
                                 current_info.get('relative_path', '')))[-1]

            edit_image_info = self.component_names.edit_dataset.format(
                ret_edit_image_height, ret_edit_image_width,
                ret_edit_image_format)

            return (gr.Gallery(value=image_list, visible=True),
                    gr.Textbox(value=current_info.get('edit_caption', '')),
                    edit_image_info)

        self.btn_reset_edit.click(
            reset_edit,
            inputs=[create_dataset.dataset_type, create_dataset.dataset_name],
            outputs=[
                self.edit_gl_dataset_images, self.edit_caption,
                self.edit_image_info
            ],
            queue=False)

        # confirm edit upload sample
        def confirm_edit(dataset_type, dataset_name):
            dataset_type = create_dataset.get_trans_dataset_type(dataset_type)
            dataset_ins = create_dataset.dataset_dict.get(
                dataset_type, {}).get(dataset_name, None)
            if dataset_ins is None:
                return (gr.Gallery(), gr.Gallery(), gr.Textbox(), gr.Text(),
                        gr.Text(), gr.Text(),
                        self.component_names.system_log.format('None'),
                        gr.Text())
            is_flg, msg = dataset_ins.apply_changes()
            if not is_flg:
                return (gr.Gallery(), gr.Gallery(), gr.Textbox(), gr.Text(),
                        gr.Text(), gr.Text(),
                        self.component_names.system_log.format(msg), gr.Text())
            image_list = [
                os.path.join(dataset_ins.local_work_dir, v['relative_path'])
                for v in dataset_ins.data
            ]
            current_record = dataset_ins.current_record
            ret_image_height = current_record.get('height', -1)
            ret_image_width = current_record.get('width', -1)
            ret_image_format = os.path.splitext(
                current_record.get('relative_path', ''))[-1]

            image_info = self.component_names.ori_dataset.format(
                ret_image_height, ret_image_width, ret_image_format)

            if dataset_type == 'scepter_img2img':
                src_image_list = [
                    os.path.join(dataset_ins.local_work_dir,
                                 v['src_relative_path'])
                    for v in dataset_ins.data
                ]
                ret_src_image_gl = gr.Gallery(
                    value=src_image_list,
                    selected_index=dataset_ins.cursor,
                    visible=True)
            else:
                ret_src_image_gl = gr.Gallery(visible=False)

            return (gr.Gallery(value=image_list,
                               selected_index=dataset_ins.cursor),
                    ret_src_image_gl,
                    gr.Textbox(value=current_record.get('caption', '')),
                    image_info, self.component_names.system_log.format(''),
                    gr.Text(value='view'))

        self.btn_confirm_edit.click(
            confirm_edit,
            inputs=[create_dataset.dataset_type, create_dataset.dataset_name],
            outputs=[
                self.gl_dataset_images, self.src_gl_dataset_images,
                self.ori_caption, self.image_info, self.sys_log,
                self.mode_state
            ],
            queue=False)

        def preprocess_box_change(preprocess_checkbox):
            image_proc_status, caption_proc_status = False, False
            reverse_status = {
                v: id
                for id, v in enumerate(self.component_names.preprocess_choices)
            }
            for value in preprocess_checkbox:
                hit_status = reverse_status[value]
                if hit_status == 0:
                    image_proc_status = True
                elif hit_status == 1:
                    caption_proc_status = True
            return (
                # gr.Column(visible=image_proc_status), gr.Column(
                # visible=caption_proc_status)
                gr.Column(visible=image_proc_status or caption_proc_status),
                gr.Row(visible=image_proc_status),
                gr.Row(visible=caption_proc_status))

        self.preprocess_checkbox.change(preprocess_box_change,
                                        inputs=[self.preprocess_checkbox],
                                        outputs=[
                                            self.preprocess_panel,
                                            self.image_preprocess_panel,
                                            self.caption_preprocess_panel
                                        ],
                                        queue=False)

        def preprocess_image(mode_state, preprocess_method, upload_image,
                             upload_src_image, upload_caption, height_ratio,
                             width_ratio, dataset_type, dataset_name):
            dataset_type = create_dataset.get_trans_dataset_type(dataset_type)
            dataset_ins = create_dataset.dataset_dict.get(
                dataset_type, {}).get(dataset_name, None)
            if dataset_ins is None:
                return (gr.Gallery(), gr.Gallery(), '', gr.Image(), gr.Image(),
                        '', '', self.component_names.system_log.format('None'))
            if hasattr(manager, 'inference'):
                for k, v in manager.inference.pipe_manager.pipeline_level_modules.items(
                ):
                    if hasattr(v, 'dynamic_unload'):
                        v.dynamic_unload(name='all')
            processor_ins = self.processors_manager.get_processor(
                'image', preprocess_method)
            if processor_ins is None:
                sys_log = 'Current processor is illegal'
                return (gr.Gallery(), gr.Gallery(), '', gr.Image(),
                        gr.Image(), '', '',
                        self.component_names.system_log.format(sys_log))
            is_flag, msg = processor_ins.load_model()
            if not is_flag:
                sys_log = f'Load processor failed: {msg}'
                return (gr.Gallery(), gr.Gallery(), '', gr.Image(),
                        gr.Image(), '', '',
                        self.component_names.system_log.format(sys_log))
            if mode_state == 'edit':
                save_folders = 'edit_images'
                os.makedirs(os.path.join(dataset_ins.meta['local_work_dir'],
                                         save_folders),
                            exist_ok=True)

                edit_index_list = dataset_ins.edit_list
                for index in edit_index_list:
                    one_data = dataset_ins.data[index]
                    # process target image
                    relative_image_path = one_data.get(
                        'edit_relative_path', one_data['relative_path'])
                    file_name, surfix = os.path.splitext(relative_image_path)

                    input_image = os.path.join(
                        dataset_ins.meta['local_work_dir'],
                        relative_image_path)
                    output_image = processor_ins(
                        Image.open(input_image).convert('RGB'),
                        height_ratio=height_ratio,
                        width_ratio=width_ratio)

                    now_time = int(time.time())
                    save_file_name = f'{os.path.basename(file_name)}_{now_time}'
                    save_relative_image_path = os.path.join(
                        save_folders, f'{get_md5(save_file_name)}{surfix}')
                    save_image_path = os.path.join(
                        dataset_ins.meta['work_dir'], save_relative_image_path)
                    local_save_image_math = os.path.join(
                        dataset_ins.meta['local_work_dir'],
                        save_relative_image_path)
                    output_image.save(local_save_image_math)
                    FS.put_object_from_local_file(local_save_image_math,
                                                  save_image_path)

                    dataset_ins.data[index][
                        'edit_relative_path'] = save_relative_image_path
                    dataset_ins.data[index][
                        'edit_image_path'] = save_image_path
                    dataset_ins.data[index]['edit_width'] = output_image.size[
                        0]
                    dataset_ins.data[index]['edit_height'] = output_image.size[
                        1]

                    if dataset_type == 'scepter_img2img':
                        # process target image
                        src_relative_image_path = one_data.get(
                            'edit_src_relative_path',
                            one_data['src_relative_path'])
                        file_name, surfix = os.path.splitext(
                            src_relative_image_path)

                        input_image = os.path.join(
                            dataset_ins.meta['local_work_dir'],
                            src_relative_image_path)
                        output_image = processor_ins(
                            Image.open(input_image).convert('RGB'),
                            height_ratio=height_ratio,
                            width_ratio=width_ratio)

                        now_time = int(time.time())
                        save_file_name = f'{os.path.basename(file_name)}_{now_time}'
                        save_src_relative_image_path = os.path.join(
                            save_folders, f'{get_md5(save_file_name)}{surfix}')
                        save_src_image_path = os.path.join(
                            dataset_ins.meta['work_dir'],
                            save_src_relative_image_path)
                        local_save_src_image_math = os.path.join(
                            dataset_ins.meta['local_work_dir'],
                            save_src_relative_image_path)
                        output_image.save(local_save_src_image_math)
                        FS.put_object_from_local_file(
                            local_save_src_image_math, save_src_image_path)

                        dataset_ins.data[index][
                            'edit_src_relative_path'] = save_src_relative_image_path
                        dataset_ins.data[index][
                            'edit_src_image_path'] = save_src_image_path
                        dataset_ins.data[index][
                            'edit_src_width'] = output_image.size[0]
                        dataset_ins.data[index][
                            'edit_src_height'] = output_image.size[1]

                dataset_ins.update_dataset()
                image_list = [
                    os.path.join(
                        dataset_ins.meta['local_work_dir'],
                        v.get('edit_relative_path', v['relative_path']))
                    for v in dataset_ins.edit_samples
                ]
                if len(edit_index_list) > 0:
                    current_info = dataset_ins.data[edit_index_list[
                        dataset_ins.edit_cursor]]

                    ret_edit_image_width = current_info.get(
                        'edit_width', current_info.get('width', -1))
                    ret_edit_image_height = current_info.get(
                        'edit_height', current_info.get('height', -1))
                    ret_edit_image_format = os.path.splitext(
                        current_info.get('edit_relative_path',
                                         current_info.get('relative_path',
                                                          '')))[-1]

                    edit_image_info = self.component_names.edit_dataset.format(
                        ret_edit_image_height, ret_edit_image_width,
                        ret_edit_image_format)

                else:
                    edit_image_info = ''

                ret_image_gallery = gr.Gallery(value=image_list)

                if dataset_type == 'scepter_img2img':
                    src_image_list = [
                        os.path.join(
                            dataset_ins.meta['local_work_dir'],
                            v.get('edit_src_relative_path',
                                  v['src_relative_path']))
                        for v in dataset_ins.edit_samples
                    ]
                    ret_src_image_gallery = gr.Gallery(value=src_image_list)
                else:
                    ret_src_image_gallery = gr.Gallery()

                ret_upload_image = gr.Image()
                ret_src_upload_image = gr.Image()
                ret_upload_image_info = ''
                ret_upload_src_image_info = gr.Markdown(visible=False)
            elif mode_state == 'add':
                if isinstance(upload_image, dict):
                    image = upload_image['image']
                else:
                    image = upload_image
                target_image = processor_ins(image.convert('RGB'),
                                             height_ratio=height_ratio,
                                             width_ratio=width_ratio)
                w, h = target_image.size
                ret_image_gallery = gr.Gallery()
                ret_src_image_gallery = gr.Gallery()
                ret_upload_image = gr.Image(target_image)

                ret_upload_image_info = self.component_names.upload_image_info.format(
                    h, w)
                if dataset_type == 'scepter_img2img':
                    if isinstance(upload_src_image, dict):
                        src_image = upload_src_image['image']
                    else:
                        src_image = upload_src_image
                    src_target_image = processor_ins(src_image.convert('RGB'),
                                                     height_ratio=height_ratio,
                                                     width_ratio=width_ratio)

                    ret_src_upload_image = gr.Image(src_target_image)
                    src_w, src_h = src_target_image.size
                    ret_upload_src_image_info = gr.Markdown(
                        value=self.component_names.upload_src_image_info.
                        format(src_h, src_w),
                        visible=True)
                else:
                    ret_src_upload_image = gr.Image()
                    ret_upload_src_image_info = gr.Markdown(visible=False)
                edit_image_info = ''
            else:
                ret_image_gallery = gr.Gallery()
                ret_src_image_gallery = gr.Gallery()
                ret_upload_image = gr.Image()
                ret_src_upload_image = gr.Image()
                ret_upload_image_info = ''
                ret_upload_src_image_info = gr.Markdown()
                edit_image_info = ''

            is_flag, msg = processor_ins.unload_model()
            if not is_flag:
                sys_log = f'Unoad processor failed: {msg}'
                return (ret_image_gallery, edit_image_info, ret_upload_image,
                        ret_src_upload_image, ret_upload_image_info,
                        self.component_names.system_log.format(sys_log))
            return (ret_image_gallery, ret_src_image_gallery, edit_image_info,
                    ret_upload_image, ret_src_upload_image,
                    ret_upload_image_info, ret_upload_src_image_info,
                    self.component_names.system_log.format(''))

        self.image_preprocess_btn.click(
            preprocess_image,
            inputs=[
                self.mode_state, self.image_preprocess_method,
                self.upload_image, self.upload_src_image, self.upload_caption,
                self.height_ratio, self.width_ratio,
                create_dataset.dataset_type, create_dataset.dataset_name
            ],
            outputs=[
                self.edit_gl_dataset_images, self.edit_src_gl_dataset_images,
                self.edit_image_info, self.upload_image, self.upload_src_image,
                self.upload_image_info, self.upload_src_image_info,
                self.sys_log
            ],
            queue=False)

        def image_preprocess_method_change(image_preprocess_method):
            image_processor_ins = self.processors_manager.get_processor(
                'image', image_preprocess_method)
            height_ratio = image_processor_ins.system_para.get(
                'HEIGHT_RATIO', {})
            ret_height_ratio = gr.Slider(minimum=height_ratio.get('MIN', 1),
                                         maximum=height_ratio.get('MAX', 10),
                                         step=height_ratio.get('STEP', 1),
                                         value=height_ratio.get('VALUE', 1),
                                         visible=len(height_ratio) > 0,
                                         interactive=True)
            width_ratio = image_processor_ins.system_para.get(
                'WIDTH_RATIO', {})
            ret_width_ratio = gr.Slider(minimum=width_ratio.get('MIN', 1),
                                        maximum=width_ratio.get('MAX', 10),
                                        step=width_ratio.get('STEP', 1),
                                        value=width_ratio.get('VALUE', 1),
                                        visible=len(width_ratio) > 0,
                                        interactive=True)
            return (ret_height_ratio, ret_width_ratio)

        self.image_preprocess_method.change(
            image_preprocess_method_change,
            inputs=[self.image_preprocess_method],
            outputs=[self.height_ratio, self.width_ratio],
            queue=False)

        def caption_preprocess_method_change(caption_preprocess_method):
            processor_ins = self.processors_manager.get_processor(
                'caption', caption_preprocess_method)
            if processor_ins is None:
                return (gr.Text(), gr.Text(), gr.Dropdown(),
                        self.component_names.system_log.format(
                            'Load processor failed, processor is None.'))
            language_choice = processor_ins.get_language_choice
            language_default = processor_ins.get_language_default
            return (gr.Text(value=processor_ins.use_device),
                    gr.Text(value=f'{processor_ins.use_memory}M'),
                    gr.Dropdown(choices=language_choice,
                                value=language_default),
                    self.component_names.system_log.format(''))

        self.caption_preprocess_method.change(
            caption_preprocess_method_change,
            inputs=[self.caption_preprocess_method],
            outputs=[
                self.caption_use_device, self.caption_use_memory,
                self.caption_language
            ],
            queue=False)

        def caption_language_change(caption_language,
                                    caption_preprocess_method):
            processor_ins = self.processors_manager.get_processor(
                'caption', caption_preprocess_method)
            para = processor_ins.get_para_by_language(caption_language)
            system_prompt = para.get('PROMPT', '')
            ret_system_prompt = gr.Text(value=system_prompt,
                                        visible='PROMPT' in para)

            max_new_tokens = para.get('MAX_NEW_TOKENS', {})
            ret_max_new_tokens = gr.Slider(
                minimum=max_new_tokens.get('MIN', 0),
                maximum=max_new_tokens.get('MAX', 1.0),
                step=max_new_tokens.get('STEP', 0.1),
                value=max_new_tokens.get('VALUE', 0.1),
                visible=len(max_new_tokens) > 0)
            min_new_tokens = para.get('MIN_NEW_TOKENS', {})
            ret_min_new_tokens = gr.Slider(
                minimum=min_new_tokens.get('MIN', 0),
                maximum=min_new_tokens.get('MAX', 1.0),
                step=min_new_tokens.get('STEP', 0.1),
                value=min_new_tokens.get('VALUE', 0.1),
                visible=len(min_new_tokens) > 0)

            num_beams = para.get('NUM_BEAMS', {})
            ret_num_beams = gr.Slider(minimum=num_beams.get('MIN', 0),
                                      maximum=num_beams.get('MAX', 1.0),
                                      step=num_beams.get('STEP', 0.1),
                                      value=num_beams.get('VALUE', 0.1),
                                      visible=len(num_beams) > 0)

            repetition_penalty = para.get('REPETITION_PENALTY', {})
            ret_repetition_penalty = gr.Slider(
                minimum=repetition_penalty.get('MIN', 0),
                maximum=repetition_penalty.get('MAX', 1.0),
                step=repetition_penalty.get('STEP', 0.1),
                value=repetition_penalty.get('VALUE', 0.1),
                visible=len(repetition_penalty) > 0)

            temperature = para.get('TEMPERATURE', {})
            ret_temperature = gr.Slider(minimum=temperature.get('MIN', 0),
                                        maximum=temperature.get('MAX', 1.0),
                                        step=temperature.get('STEP', 0.1),
                                        value=temperature.get('VALUE', 0.1),
                                        visible=len(temperature) > 0)

            return (ret_system_prompt, ret_max_new_tokens, ret_min_new_tokens,
                    ret_num_beams, ret_repetition_penalty, ret_temperature)

        self.caption_language.change(
            caption_language_change,
            inputs=[self.caption_language, self.caption_preprocess_method],
            outputs=[
                self.sys_prompt, self.max_new_tokens, self.min_new_tokens,
                self.num_beams, self.repetition_penalty, self.temperature
            ],
            queue=False)

        def preprocess_caption(mode_state, preprocess_method, sys_prompt,
                               max_new_tokens, min_new_tokens, num_beams,
                               repetition_penalty, temperature,
                               caption_update_mode, upload_image,
                               upload_src_image, upload_caption, dataset_type,
                               dataset_name):

            reverse_update_mode = {
                v: idx
                for idx, v in enumerate(
                    self.component_names.caption_update_choices)
            }

            update_mode = reverse_update_mode.get(caption_update_mode, -1)

            processor_ins = self.processors_manager.get_processor(
                'caption', preprocess_method)
            if processor_ins is None:
                sys_log = 'Current processor is illegal'
                return gr.Textbox(), gr.Textbox(
                ), self.component_names.system_log.format(sys_log)

            is_flag, msg = processor_ins.load_model()
            if not is_flag:
                sys_log = f'Load processor failed: {msg}'
                return gr.Textbox(), gr.Textbox(
                ), self.component_names.system_log.format(sys_log)
            dataset_type = create_dataset.get_trans_dataset_type(dataset_type)
            dataset_ins = create_dataset.dataset_dict.get(
                dataset_type, {}).get(dataset_name, None)
            if dataset_ins is None:
                return gr.Textbox(), gr.Textbox(
                ), self.component_names.system_log.format('None')
            if mode_state == 'edit':
                edit_index_list = dataset_ins.edit_list
                for index in edit_index_list:
                    one_data = dataset_ins.data[index]
                    relative_image_path = one_data.get(
                        'edit_relative_path', one_data['relative_path'])
                    src_image = os.path.join(
                        dataset_ins.meta['local_work_dir'],
                        relative_image_path)
                    response = processor_ins(
                        src_image,
                        prompt=sys_prompt,
                        max_new_tokens=max_new_tokens,
                        min_new_tokens=min_new_tokens,
                        num_beams=num_beams,
                        repetition_penalty=repetition_penalty,
                        temperature=temperature)

                    if dataset_type == 'scepter_img2img':
                        relative_image_path = one_data.get(
                            'edit_relative_path',
                            one_data['src_relative_path'])
                        src_image = os.path.join(
                            dataset_ins.meta['local_work_dir'],
                            relative_image_path)
                        src_response = processor_ins(
                            src_image,
                            prompt=sys_prompt,
                            max_new_tokens=max_new_tokens,
                            min_new_tokens=min_new_tokens,
                            num_beams=num_beams,
                            repetition_penalty=repetition_penalty,
                            temperature=temperature)
                        response = 'src: ' + src_response + ' target: ' + response

                    if update_mode == 0:
                        if len(dataset_ins.data[index]['edit_caption']) > 0:
                            dataset_ins.data[index]['edit_caption'] += ';'
                        dataset_ins.data[index]['edit_caption'] += response
                    elif update_mode == 1:
                        dataset_ins.data[index]['edit_caption'] = response

                dataset_ins.update_dataset()
                current_info = dataset_ins.current_record
                ret_edit_caption = gr.Textbox(value=current_info.get(
                    'edit_caption', ''),
                                              visible=True)
                ret_upload_caption = gr.Textbox()
            elif mode_state == 'add':
                save_folder = os.path.join(dataset_ins.local_work_dir, 'cache')
                os.makedirs(save_folder, exist_ok=True)
                if isinstance(upload_image, dict):
                    image = upload_image['image']
                else:
                    image = upload_image
                w, h = image.size
                image_path = os.path.join(
                    save_folder, f'{imagehash.phash(image)}_{w}_{h}.png')
                if not os.path.exists(image_path):
                    image.save(image_path)
                response = processor_ins(image_path,
                                         prompt=sys_prompt,
                                         max_new_tokens=max_new_tokens,
                                         min_new_tokens=min_new_tokens,
                                         num_beams=num_beams,
                                         repetition_penalty=repetition_penalty,
                                         temperature=temperature)

                if dataset_type == 'scepter_img2img':
                    if isinstance(upload_src_image, dict):
                        src_image = upload_src_image['image']
                    else:
                        src_image = upload_src_image
                    w, h = src_image.size
                    src_image_path = os.path.join(
                        save_folder,
                        f'{imagehash.phash(src_image)}_{w}_{h}.png')
                    if not os.path.exists(src_image_path):
                        src_image.save(src_image_path)
                    src_response = processor_ins(
                        src_image_path,
                        prompt=sys_prompt,
                        max_new_tokens=max_new_tokens,
                        min_new_tokens=min_new_tokens,
                        num_beams=num_beams,
                        repetition_penalty=repetition_penalty,
                        temperature=temperature)
                    response = 'src: ' + src_response + ' target: ' + response

                ret_edit_caption = gr.Textbox()
                if update_mode == 0:
                    if len(upload_caption) > 0:
                        upload_caption += ';'
                    upload_caption += response
                elif update_mode == 1:
                    upload_caption = response
                ret_upload_caption = gr.Textbox(value=upload_caption)
            else:
                ret_edit_caption = gr.Textbox()
                ret_upload_caption = gr.Textbox()

            is_flag, msg = processor_ins.unload_model()
            if not is_flag:
                sys_log = f'Unoad processor failed: {msg}'
                return gr.Textbox(
                ), ret_upload_caption, self.component_names.system_log.format(
                    sys_log)

            return ret_edit_caption, ret_upload_caption, self.component_names.system_log.format(
                '')

        self.caption_preprocess_btn.click(
            preprocess_caption,
            inputs=[
                self.mode_state, self.caption_preprocess_method,
                self.sys_prompt, self.max_new_tokens, self.min_new_tokens,
                self.num_beams, self.repetition_penalty, self.temperature,
                self.caption_update_mode, self.upload_image,
                self.upload_src_image, self.upload_caption,
                create_dataset.dataset_type, create_dataset.dataset_name
            ],
            outputs=[self.edit_caption, self.upload_caption, self.sys_log],
            queue=False)

        def mode_state_change(mode_state, dataset_type, dataset_name):
            # default is editing current sample
            dataset_type = create_dataset.get_trans_dataset_type(dataset_type)
            if mode_state == 'edit':
                dataset_ins = create_dataset.dataset_dict.get(
                    dataset_type, {}).get(dataset_name, None)
                if dataset_ins is None or len(dataset_ins) < 1:
                    return (
                        gr.Gallery(),
                        gr.Gallery(),
                        gr.Column(),
                        gr.Row(visible=True),
                        gr.Column(),
                        gr.Column(),
                        gr.Image(),
                        gr.Column(),
                        '',
                        gr.Markdown(),
                        gr.Textbox(),
                        gr.CheckboxGroup(),
                        gr.Column(),
                        # gr.Column(),
                        gr.Row(),
                        # gr.Column(),
                        gr.Row(),
                        gr.Column(),
                        gr.Gallery(),
                        gr.Gallery(),
                        gr.Column(),
                        gr.Dropdown(),
                        gr.Dropdown(value=[]),
                        self.component_names.system_log.format(
                            self.component_names.illegal_blank_dataset))
                dataset_ins.set_edit_range(str(dataset_ins.cursor + 1))
                image_list = [
                    os.path.join(
                        dataset_ins.meta['local_work_dir'],
                        v.get('edit_relative_path', v['relative_path']))
                    for v in dataset_ins.edit_samples
                ]
                selected_index = dataset_ins.edit_index_from_cursor(
                    dataset_ins.cursor)
                dataset_ins.edit_cursor = selected_index
                current_info = dataset_ins.current_record
                ret_edit_caption = gr.Textbox(value=current_info.get(
                    'edit_caption', ''),
                                              visible=True)

                if dataset_type == 'scepter_img2img':
                    src_image_list = [
                        os.path.join(
                            dataset_ins.meta['local_work_dir'],
                            v.get('edit_src_relative_path',
                                  v['src_relative_path']))
                        for v in dataset_ins.edit_samples
                    ]
                    if dataset_ins.edit_cursor >= 0:
                        ret_edit_src_gallery = gr.Gallery(
                            label=dataset_name,
                            value=src_image_list,
                            selected_index=selected_index,
                            visible=True)
                    else:
                        ret_edit_src_gallery = gr.Gallery(label=dataset_name,
                                                          value=src_image_list,
                                                          selected_index=None,
                                                          visible=True)
                    ret_edit_src_panel = gr.Column(visible=True)
                else:
                    ret_edit_src_gallery = gr.Gallery(visible=False)
                    ret_edit_src_panel = gr.Column(visible=False)
                return (
                    gr.Gallery(),
                    gr.Gallery(),
                    gr.Column(),
                    gr.Row(visible=True),
                    gr.Column(visible=True),
                    gr.Column(visible=False),
                    gr.Image(),
                    gr.Column(),
                    '',
                    gr.Markdown(visible=False),
                    gr.Textbox(),
                    gr.CheckboxGroup(value=None),
                    gr.Column(visible=False),
                    # gr.Column(visible=False),
                    gr.Row(visible=False),
                    # gr.Column(visible=False),
                    gr.Row(visible=False),
                    gr.Column(visible=True),
                    gr.Gallery(value=image_list,
                               selected_index=selected_index,
                               visible=True),
                    ret_edit_caption,
                    ret_edit_src_gallery,
                    ret_edit_src_panel,
                    gr.Dropdown(
                        # choices=self.component_names.range_mode_name,
                        # value=self.component_names.range_mode_name[0]
                    ),
                    gr.Dropdown(value=[]),
                    self.component_names.system_log.format(''))
            elif mode_state == 'add':
                return (
                    gr.Gallery(),
                    gr.Gallery(),
                    gr.Column(),
                    gr.Row(visible=False),
                    gr.Column(visible=False),
                    gr.Column(visible=True),
                    gr.Image(),
                    gr.Column(visible=True) if dataset_type
                    == 'scepter_img2img' else gr.Column(visible=False),
                    '',
                    gr.Markdown(visible=True) if dataset_type
                    == 'scepter_img2img' else gr.Markdown(visible=False),
                    gr.Textbox(value=''),
                    gr.CheckboxGroup(),
                    gr.Column(visible=True),
                    # gr.Column(visible=True),
                    gr.Row(visible=True),
                    # gr.Column(visible=True),
                    gr.Row(visible=True),
                    gr.Column(visible=False),
                    gr.Gallery(visible=False),
                    gr.Textbox(),
                    gr.Gallery(visible=False),
                    gr.Column(visible=False),
                    gr.Dropdown(
                        # choices=self.component_names.range_mode_name,
                        # value=self.component_names.range_mode_name[0]
                    ),
                    gr.Dropdown(value=[]),
                    self.component_names.system_log.format(''))
            else:
                dataset_ins = create_dataset.dataset_dict.get(
                    dataset_type, {}).get(dataset_name, None)
                if dataset_ins is None:
                    return (
                        gr.Gallery(),
                        gr.Gallery(),
                        gr.Column(),
                        gr.Row(visible=True),
                        gr.Column(),
                        gr.Column(),
                        gr.Image(),
                        gr.Column(),
                        '',
                        gr.Markdown(),
                        gr.Textbox(),
                        gr.CheckboxGroup(),
                        gr.Column(),
                        # gr.Column(),
                        gr.Row(),
                        # gr.Column(),
                        gr.Row(),
                        gr.Column(),
                        gr.Gallery(),
                        gr.Textbox(),
                        gr.Gallery(),
                        gr.Column(),
                        gr.Dropdown(),
                        gr.Dropdown(value=[]),
                        self.component_names.system_log.format(
                            self.component_names.illegal_blank_dataset))
                cursor = dataset_ins.cursor
                image_list = [
                    os.path.join(dataset_ins.meta['local_work_dir'],
                                 v['relative_path']) for v in dataset_ins.data
                ]
                if dataset_type == 'scepter_img2img':
                    src_image_list = [
                        os.path.join(dataset_ins.meta['local_work_dir'],
                                     v['src_relative_path'])
                        for v in dataset_ins.data
                    ]
                    if cursor >= 0:
                        ret_src_gallery = gr.Gallery(label=dataset_name,
                                                     value=src_image_list,
                                                     selected_index=cursor,
                                                     visible=True)
                    else:
                        ret_src_gallery = gr.Gallery(label=dataset_name,
                                                     value=src_image_list,
                                                     selected_index=None,
                                                     visible=True)
                    ret_src_panel = gr.Column(visible=True)
                else:
                    ret_src_gallery = gr.Gallery(visible=False)
                    ret_src_panel = gr.Column(visible=False)

                if cursor >= 0:
                    ret_gl_gallery = gr.Gallery(label=dataset_name,
                                                value=image_list,
                                                selected_index=cursor)
                else:
                    ret_gl_gallery = gr.Gallery(label=dataset_name,
                                                value=image_list,
                                                selected_index=None)

                return (
                    ret_gl_gallery,
                    ret_src_gallery,
                    ret_src_panel,
                    gr.Row(visible=False),
                    gr.Column(visible=False),
                    gr.Column(visible=False),
                    gr.Image(),
                    gr.Column(),
                    '',
                    gr.Markdown(visible=False),
                    gr.Textbox(value=''),
                    gr.CheckboxGroup(value=None),
                    gr.Column(visible=False),
                    # gr.Column(visible=False),
                    gr.Row(visible=False),
                    # gr.Column(visible=False),
                    gr.Row(visible=False),
                    gr.Column(visible=False),
                    gr.Gallery(visible=False),
                    gr.Textbox(),
                    gr.Gallery(visible=False),
                    gr.Column(visible=False),
                    gr.Dropdown(
                        # choices=self.component_names.range_mode_name,
                        # value=self.component_names.range_mode_name[0]
                    ),
                    gr.Dropdown(value=[]),
                    '')

        self.mode_state.change(
            mode_state_change,
            inputs=[
                self.mode_state, create_dataset.dataset_type,
                create_dataset.dataset_name
            ],
            outputs=[
                self.gl_dataset_images, self.src_gl_dataset_images,
                self.src_panel, self.edit_setting_panel,
                self.edit_confirm_panel, self.upload_panel, self.upload_image,
                self.upload_src_image_panel, self.upload_image_info,
                self.upload_src_image_info, self.upload_caption,
                self.preprocess_checkbox, self.preprocess_panel,
                self.image_preprocess_panel, self.caption_preprocess_panel,
                self.edit_panel, self.edit_gl_dataset_images,
                self.edit_caption, self.edit_src_gl_dataset_images,
                self.edit_src_panel, self.range_mode, self.data_range,
                self.sys_log
            ],
            queue=False)

        def range_change(range_mode, dataset_type, dataset_name):
            hit_range_mode = range_state_trans(range_mode)
            if hit_range_mode == 2:
                return (gr.Dropdown(visible=True), gr.Gallery(), gr.Gallery())
            else:
                dataset_type = create_dataset.get_trans_dataset_type(
                    dataset_type)
                dataset_ins = create_dataset.dataset_dict.get(
                    dataset_type, {}).get(dataset_name, None)
                if dataset_ins is None:
                    return (gr.Dropdown(), gr.Gallery(), gr.Gallery())
                if hit_range_mode == 1:
                    dataset_ins.set_edit_range(-1)
                else:
                    dataset_ins.set_edit_range(str(dataset_ins.cursor + 1))
                image_list = [
                    os.path.join(
                        dataset_ins.meta['local_work_dir'],
                        v.get('edit_relative_path', v['relative_path']))
                    for v in dataset_ins.edit_samples
                ]
                selected_index = dataset_ins.edit_index_from_cursor(
                    dataset_ins.cursor)
                dataset_ins.edit_cursor = selected_index
                if dataset_type == 'scepter_img2img':
                    src_image_list = [
                        os.path.join(
                            dataset_ins.meta['local_work_dir'],
                            v.get('edit_src_relative_path',
                                  v['src_relative_path']))
                        for v in dataset_ins.edit_samples
                    ]
                    if dataset_ins.edit_cursor >= 0:
                        ret_edit_src_gallery = gr.Gallery(
                            label=dataset_name,
                            value=src_image_list,
                            selected_index=selected_index)
                    else:
                        ret_edit_src_gallery = gr.Gallery(label=dataset_name,
                                                          value=src_image_list,
                                                          selected_index=None)
                else:
                    ret_edit_src_gallery = gr.Gallery()
                return (gr.Dropdown(visible=False),
                        gr.Gallery(value=image_list,
                                   selected_index=selected_index,
                                   visible=True), ret_edit_src_gallery)

        self.range_mode.change(range_change,
                               inputs=[
                                   self.range_mode,
                                   create_dataset.dataset_type,
                                   create_dataset.dataset_name
                               ],
                               outputs=[
                                   self.data_range,
                                   self.edit_gl_dataset_images,
                                   self.edit_src_gl_dataset_images
                               ],
                               queue=False)

        def set_range(data_range, dataset_type, dataset_name):
            if len(data_range) == 0:
                return (gr.Gallery(), gr.Gallery(), gr.Gallery(), gr.Textbox(),
                        gr.Text(), gr.Textbox(), gr.Text(), gr.Text(),
                        self.component_names.system_log.format(''))
            data_range = ','.join(data_range)
            dataset_type = create_dataset.get_trans_dataset_type(dataset_type)
            dataset_ins = create_dataset.dataset_dict.get(
                dataset_type, {}).get(dataset_name, None)
            if dataset_ins is None:
                return (gr.Gallery(), gr.Gallery(), gr.Gallery(), gr.Textbox(),
                        gr.Text(), gr.Textbox(), gr.Text(), gr.Text(), 'None')
            flg, msg = dataset_ins.set_edit_range(data_range)
            if not flg:
                sys_log = self.component_names.system_log.format(msg)
                return (gr.Gallery(), gr.Gallery(), gr.Gallery(), gr.Textbox(),
                        gr.Text(), gr.Textbox(), gr.Text(), gr.Text(), sys_log)
            image_list = [
                os.path.join(dataset_ins.meta['local_work_dir'],
                             v.get('edit_relative_path', v['relative_path']))
                for v in dataset_ins.edit_samples
            ]

            selected_index = 0 if len(image_list) > 0 else -1
            if selected_index >= 0:
                cursor = dataset_ins.cursor_from_edit_index(selected_index)
                dataset_ins.set_cursor(cursor)
                current_info = dataset_ins.current_record
                all_number = len(dataset_ins)

                ret_image_gl = gr.Gallery(selected_index=dataset_ins.cursor) if dataset_ins.cursor >= 0 \
                    else gr.Gallery(value=[], selected_index=None)
                ret_edit_image_gl = gr.Gallery(value=image_list,
                                               selected_index=selected_index,
                                               visible=True)
                if dataset_type == 'scepter_img2img':
                    src_image_list = [
                        os.path.join(
                            dataset_ins.meta['local_work_dir'],
                            v.get('edit_src_relative_path',
                                  v['src_relative_path']))
                        for v in dataset_ins.edit_samples
                    ]
                    ret_edit_src_image_gl = gr.Gallery(
                        value=src_image_list,
                        selected_index=selected_index,
                        visible=True)
                else:
                    ret_edit_src_image_gl = gr.Gallery()

                ret_caption = gr.Textbox(value=current_info.get('caption', ''),
                                         visible=True)
                ret_edit_caption = gr.Textbox(value=current_info.get(
                    'edit_caption', ''),
                                              visible=True)

                ret_image_height = current_info.get('height', -1)
                ret_image_width = current_info.get('width', -1)
                ret_image_format = os.path.splitext(
                    current_info.get('relative_path', ''))[-1]

                image_info = self.component_names.ori_dataset.format(
                    ret_image_height, ret_image_width, ret_image_format)

                ret_edit_image_width = current_info.get(
                    'edit_width', current_info.get('width', -1))
                ret_edit_image_height = current_info.get(
                    'edit_height', current_info.get('height', -1))
                ret_edit_image_format = os.path.splitext(
                    current_info.get('edit_relative_path',
                                     current_info.get('relative_path',
                                                      '')))[-1]

                edit_image_info = self.component_names.edit_dataset.format(
                    ret_edit_image_height, ret_edit_image_width,
                    ret_edit_image_format)

                ret_info = gr.Text(
                    value=f'{dataset_ins.cursor + 1}/{all_number}')
            else:
                ret_image_gl = gr.Gallery()
                ret_edit_image_gl = gr.Gallery()
                ret_edit_src_image_gl = gr.Gallery()
                ret_caption = gr.Textbox()
                ret_edit_caption = gr.Textbox()

                image_info = ''
                edit_image_info = ''
                ret_info = gr.Text()
            return (ret_image_gl, ret_edit_image_gl, ret_edit_src_image_gl,
                    ret_caption, image_info, ret_edit_caption, edit_image_info,
                    ret_info, self.component_names.system_log.format(''))

        self.data_range.change(
            set_range,
            inputs=[
                self.data_range, create_dataset.dataset_type,
                create_dataset.dataset_name
            ],
            outputs=[
                self.gl_dataset_images, self.edit_gl_dataset_images,
                self.edit_src_gl_dataset_images, self.ori_caption,
                self.image_info, self.edit_caption, self.edit_image_info,
                self.info, self.sys_log
            ],
            queue=False)

        def delete_file(dataset_type, dataset_name):
            dataset_type = create_dataset.get_trans_dataset_type(dataset_type)
            dataset_ins = create_dataset.dataset_dict.get(
                dataset_type, {}).get(dataset_name, None)
            if dataset_ins is None or len(dataset_ins) < 1:
                return (gr.Gallery(), gr.Checkbox(), gr.Row(visible=True),
                        self.component_names.system_log.format(
                            self.component_names.delete_blank_dataset))
            dataset_ins.delete_record()
            if len(dataset_ins) > 0:
                image_list = [
                    os.path.join(dataset_ins.local_work_dir,
                                 v['relative_path']) for v in dataset_ins.data
                ]
                ret_gallery = gr.Gallery(value=image_list,
                                         selected_index=dataset_ins.cursor)
                ret_state = gr.Checkbox(value=False)
            else:
                ret_gallery = gr.Gallery(value=[], selected_index=None)
                ret_state = gr.Checkbox(value=True)
            return (ret_gallery, ret_state, gr.Row(visible=True),
                    self.component_names.system_log.format(''))

        self.delete_button.click(delete_file,
                                 inputs=[
                                     create_dataset.dataset_type,
                                     create_dataset.user_dataset_name
                                 ],
                                 outputs=[
                                     self.gl_dataset_images,
                                     self.src_dataset_state,
                                     self.edit_setting_panel, self.sys_log
                                 ],
                                 queue=False)

        def src_dataset_change(src_dataset_state, dataset_type, dataset_name):
            dataset_type = create_dataset.get_trans_dataset_type(dataset_type)
            dataset_ins = create_dataset.dataset_dict.get(
                dataset_type, {}).get(dataset_name, None)
            if dataset_ins is None:
                return (gr.Gallery(), gr.Row(visible=True),
                        self.component_names.system_log.format(
                            self.component_names.illegal_blank_dataset))
            if src_dataset_state:
                if dataset_type == 'scepter_img2img':
                    src_image_list = [
                        os.path.join(dataset_ins.meta['local_work_dir'],
                                     v['src_relative_path'])
                        for v in dataset_ins.data
                    ]
                    if len(dataset_ins) > 0:
                        ret_src_image_gl = gr.Gallery(
                            value=src_image_list,
                            selected_index=dataset_ins.cursor,
                            visible=True)
                    else:
                        ret_src_image_gl = gr.Gallery(value=[],
                                                      selected_index=None,
                                                      visible=True)
                else:
                    ret_src_image_gl = gr.Gallery(visible=False)
            else:
                ret_src_image_gl = gr.Gallery()
            return (ret_src_image_gl, gr.Row(visible=True),
                    self.component_names.system_log.format(''))

        self.src_dataset_state.change(src_dataset_change,
                                      inputs=[
                                          self.src_dataset_state,
                                          create_dataset.dataset_type,
                                          create_dataset.user_dataset_name
                                      ],
                                      outputs=[
                                          self.src_gl_dataset_images,
                                          self.edit_setting_panel, self.sys_log
                                      ],
                                      queue=False)

        def image_upload(upload_image):
            if isinstance(upload_image, dict):
                image = upload_image['image']
            else:
                image = upload_image
            w, h = image.size
            return self.component_names.upload_image_info.format(h, w)

        self.upload_image.upload(image_upload,
                                 inputs=[self.upload_image],
                                 outputs=[self.upload_image_info],
                                 queue=False)

        def image_src_upload(upload_image):
            if isinstance(upload_image, dict):
                image = upload_image['image']
            else:
                image = upload_image
            w, h = image.size
            return gr.Markdown(
                value=self.component_names.upload_src_image_info.format(h, w),
                visible=True)

        self.upload_src_image.upload(image_src_upload,
                                     inputs=[self.upload_src_image],
                                     outputs=[self.upload_src_image_info],
                                     queue=False)

        def image_clear():
            return gr.Markdown(visible=False)

        self.upload_image.clear(image_clear,
                                inputs=[],
                                outputs=[self.upload_image_info],
                                queue=False)

        def src_image_clear():
            return ''

        self.upload_src_image.clear(image_clear,
                                    inputs=[],
                                    outputs=[self.upload_src_image_info],
                                    queue=False)

        def show_add_record_panel():
            return gr.Text(value='add')

        self.add_button.click(
            show_add_record_panel,
            inputs=[],
            outputs=[self.mode_state],
        )

        def add_file(dataset_type, dataset_name, upload_image,
                     upload_src_image, caption):
            if upload_image is None:
                return (gr.Gallery(), gr.Textbox(), '', gr.Textbox(), '',
                        gr.Text(), gr.Image(), gr.Image(), gr.Text(), '', '',
                        gr.Text())
            if isinstance(upload_image, dict):
                image = upload_image['image']
            else:
                image = upload_image

            if isinstance(upload_src_image, dict):
                src_image = upload_src_image['image']
            else:
                src_image = upload_src_image

            dataset_type = create_dataset.get_trans_dataset_type(dataset_type)
            dataset_ins = create_dataset.dataset_dict.get(
                dataset_type, {}).get(dataset_name, None)
            if dataset_ins is None:
                return (gr.Gallery(), gr.Textbox(), '', gr.Textbox(), '',
                        gr.Text(), gr.Image(), gr.Image(), gr.Text(), '', '',
                        gr.Text())
            dataset_ins.add_record(image, caption, src_image=src_image)
            return (gr.Image(value=None), gr.Image(value=None),
                    gr.Text(value=''), '', '', gr.Text(value='view'))

        self.upload_button.click(
            add_file,
            inputs=[
                create_dataset.dataset_type, create_dataset.dataset_name,
                self.upload_image, self.upload_src_image, self.upload_caption
            ],
            outputs=[
                # self.gl_dataset_images,
                # self.ori_caption,
                # self.image_info, self.edit_caption,
                # self.edit_image_info, self.info,
                self.upload_image,
                self.upload_src_image,
                self.upload_caption,
                self.upload_image_info,
                self.upload_src_image_info,
                self.mode_state
            ],
            queue=False)

        def cancel_add_file():
            return gr.Text(value='view')

        self.cancel_button.click(cancel_add_file,
                                 inputs=[],
                                 outputs=[self.mode_state],
                                 queue=False)

        def edit_caption_change(dataset_type, dataset_name, edit_caption):
            dataset_type = create_dataset.get_trans_dataset_type(dataset_type)
            dataset_ins = create_dataset.dataset_dict.get(
                dataset_type, {}).get(dataset_name, None)
            if dataset_ins is None:
                return
            dataset_ins.edit_caption(edit_caption)

        self.edit_caption.change(edit_caption_change,
                                 inputs=[
                                     create_dataset.dataset_type,
                                     create_dataset.dataset_name,
                                     self.edit_caption
                                 ],
                                 queue=False)
