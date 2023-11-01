import argparse
import copy
import io
import math
import operator
import os
import sys
from typing import Literal
import PIL.Image
import gltflib
import numpy as np
import struct

v2f = tuple[float, float]
v3f = tuple[float, float, float]
v4f = tuple[float, float, float, float]

bs_blinx = [
    (0x00ac2e2c, 0x00ae62a4),
    (0x00cfea44, 0x00ae62a4),
    (0x00d0c954, 0x00b0915c),
    (0x00d1afbc, 0x00ae62a4),
    (0x00d290f4, 0x00ae62a4),
    (0x00d371fc, 0x00ae62a4),
    (0x00d4511c, 0x00ae62a4),
    (0x00d531ac, 0x00ae62a4),
    (0x00d610f4, 0x00ae62a4),
    (0x00d6f094, 0x00ae62a4),
    (0x00d7cfec, 0x00ae62a4),
    (0x00f07954, 0x00ae62a4),
]

bs_001d4bd8 = [
    (0x00ac35bc, 0x00ae6a34),
    (0x00abddac, 0x00ae127c),
    (0x00abd9d8, 0x00ae0ea8),
    (0x00abd5cc, 0x00ae0a9c),
    (0x00abd1f8, 0x00ae06c8),
    (0x00ab3780, 0x00ad69d0),
    (0x00ab3528, 0x00ad6778),
    (0x00ab31f4, 0x00ad6444),
    (0x00ab2e34, 0x00ad6084),
    (0x00cff1d4, 0x00ae6a34),
    (0x00cf9a0c, 0x00ae127c),
    (0x00cf9638, 0x00ae0ea8),
    (0x00cf922c, 0x00ae0a9c),
    (0x00cf8e58, 0x00ae06c8),
    (0x00d0d0e4, 0x00b098ec),
    (0x00d0791c, 0x00b0410c),
    (0x00d07548, 0x00b03d38),
    (0x00d0713c, 0x00b0392c),
    (0x00d06d68, 0x00b03558),
    (0x00d1b74c, 0x00ae6a34),
    (0x00d15f6c, 0x00ae127c),
    (0x00d15b98, 0x00ae0ea8),
    (0x00d1578c, 0x00ae0a9c),
    (0x00d153b8, 0x00ae06c8),
    (0x00d29884, 0x00ae6a34),
    (0x00d24084, 0x00ae127c),
    (0x00d23cb0, 0x00ae0ea8),
    (0x00d238a4, 0x00ae0a9c),
    (0x00d234d0, 0x00ae06c8),
    (0x00d3798c, 0x00ae6a34),
    (0x00d321bc, 0x00ae127c),
    (0x00d31de8, 0x00ae0ea8),
    (0x00d319dc, 0x00ae0a9c),
    (0x00d31608, 0x00ae06c8),
    (0x00d458ac, 0x00ae6a34),
    (0x00d400d4, 0x00ae127c),
    (0x00d3fd00, 0x00ae0ea8),
    (0x00d3f8f4, 0x00ae0a9c),
    (0x00d3f520, 0x00ae06c8),
    (0x00d5393c, 0x00ae6a34),
    (0x00d4e164, 0x00ae127c),
    (0x00d4dd90, 0x00ae0ea8),
    (0x00d4d984, 0x00ae0a9c),
    (0x00d4d5b0, 0x00ae06c8),
    (0x00d61884, 0x00ae6a34),
    (0x00d5c084, 0x00ae127c),
    (0x00d5bcb0, 0x00ae0ea8),
    (0x00d5b8a4, 0x00ae0a9c),
    (0x00d5b4d0, 0x00ae06c8),
    (0x00d6f824, 0x00ae6a34),
    (0x00d69fec, 0x00ae127c),
    (0x00d69c18, 0x00ae0ea8),
    (0x00d6980c, 0x00ae0a9c),
    (0x00d69438, 0x00ae06c8),
    (0x00d7d77c, 0x00ae6a34),
    (0x00d77f8c, 0x00ae127c),
    (0x00d77bb8, 0x00ae0ea8),
    (0x00d777ac, 0x00ae0a9c),
    (0x00d773d8, 0x00ae06c8),
    (0x00f080e4, 0x00ae6a34),
    (0x00f02904, 0x00ae127c),
    (0x00f02530, 0x00ae0ea8),
    (0x00f02124, 0x00ae0a9c),
    (0x00f01d50, 0x00ae06c8),
    (0x00209744, 0x0021e5bc),
    (0x00206df8, 0x0021bc70),
    (0x00206d20, 0x0021bb98),
    (0x002066b8, 0x0021b530),
    (0x00205d18, 0x0021ab90),
    (0x002056f0, 0x0021a568),
    (0x002050cc, 0x00219f44),
    (0x00204cf4, 0x00219b6c),
    (0x00204884, 0x002196fc),
    (0x002044ac, 0x00219324),
]

bs_001d4e30 = [
    (0x00abd1f8, 0x00ae06c8),
    (0x00ab2e34, 0x00ad6084),
    (0x00cf8e58, 0x00ae06c8),
    (0x00d06d68, 0x00b03558),
    (0x00d153b8, 0x00ae06c8),
    (0x00d234d0, 0x00ae06c8),
    (0x00d31608, 0x00ae06c8),
    (0x00d3f520, 0x00ae06c8),
    (0x00d4d5b0, 0x00ae06c8),
    (0x00d5b4d0, 0x00ae06c8),
    (0x00d69438, 0x00ae06c8),
    (0x00d773d8, 0x00ae06c8),
    (0x00f01d50, 0x00ae06c8),
    (0x002044ac, 0x00219324),
    (0x00000000, 0x00000000),
]

class UnrecognizedChunkIdentifierError(Exception):
    def __init__(self, chunk_ident, virtual_address):
        self.chunk_ident = chunk_ident
        self.message = f'Unrecognized triangle chunk identifier! ({chunk_ident:#0{4}x}) at {virtual_address:#0{10}x}'
        super().__init__(self.message)

class XBE:
    class Section:
        def __init__(self, vaddr, vsize, raddr, rsize):
            self.vaddr = vaddr
            self.vsize = vsize
            self.raddr = raddr
            self.rsize = rsize

    def __init__(self, f, sections: list[Section]):
        self.f = f
        self.sections = sections

    @staticmethod
    def from_filepath(filepath):
        f = open(filepath, 'rb')

        f.seek(0x104)
        baseaddr = struct.unpack('i', f.read(4))[0]
        
        f.seek(0x11c)
        num_sections = struct.unpack('i', f.read(4))[0]
        
        f.seek(0x120)
        vaddr_sections = struct.unpack('i', f.read(4))[0]

        sections = []
        for i in range(num_sections):
            f.seek((vaddr_sections - baseaddr) + 0x38 * i)
            
            f.seek(4, 1)
            vaddr = struct.unpack('i', f.read(4))[0]
            vsize = struct.unpack('i', f.read(4))[0]
            raddr = struct.unpack('i', f.read(4))[0]
            rsize = struct.unpack('i', f.read(4))[0]
            sections.append(XBE.Section(vaddr, vsize, raddr, rsize))

        return XBE(f, sections)

class XBEBinaryReader:
    def __init__(self, xbe: XBE) -> None:
        self.xbe = xbe

    def virtual_address_to_raw_address(self, virtual_address):
        for i in range(len(self.xbe.sections)):
            section = self.xbe.sections[i]
            if virtual_address >= section.vaddr and virtual_address < section.vaddr + section.vsize:
                return (virtual_address - section.vaddr) + section.raddr
        return None
    
    def read_bytes(self, virtual_address, n) -> bytes:
        self.xbe.f.seek(self.virtual_address_to_raw_address(virtual_address))
        return self.xbe.f.read(n)
    
    def read_i8(self, virtual_address: int) -> int:
        return struct.unpack('b', self.read_bytes(virtual_address, 1))[0]

    def read_u8(self, virtual_address: int) -> int:
        return struct.unpack('B', self.read_bytes(virtual_address, 1))[0]

    def read_i16(self, virtual_address) -> int:
        return struct.unpack('<h', self.read_bytes(virtual_address, 2))[0]

    def read_u16(self, virtual_address) -> int:
        return struct.unpack('<H', self.read_bytes(virtual_address, 2))[0]

    def read_i32(self, virtual_address) -> int:
        return struct.unpack('<i', self.read_bytes(virtual_address, 4))[0]

    def read_u32(self, virtual_address) -> int:
        return struct.unpack('<I', self.read_bytes(virtual_address, 4))[0]

    def read_f32(self, virtual_address) -> int:
        return struct.unpack('f', self.read_bytes(virtual_address, 4))[0]

    def read_v3f(self, virtual_address) -> v3f:
        return struct.unpack('3f', self.read_bytes(virtual_address, 12))

class Matrix4x4:
    def __init__(self) -> None:
        self.data = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])

    def translate(self, x, y, z):
        translation_matrix = np.array([
            [1, 0, 0, x],
            [0, 1, 0, y],
            [0, 0, 1, z],
            [0, 0, 0, 1],
        ])
        self.data = np.dot(self.data, translation_matrix)

    def rotate_x(self, r):
        sin_theta = np.sin(r)
        cos_theta = np.cos(r)

        rotation_matrix = np.array([
            [1, 0, 0, 0],
            [0, cos_theta, -sin_theta, 0],
            [0, sin_theta, cos_theta, 0],
            [0, 0, 0, 1],
        ])
        self.data = np.dot(self.data, rotation_matrix)

    def rotate_y(self, r):
        sin_theta = np.sin(r)
        cos_theta = np.cos(r)

        rotation_matrix = np.array([
            [cos_theta, 0, sin_theta, 0],
            [0, 1, 0, 0],
            [-sin_theta, 0, cos_theta, 0],
            [0, 0, 0, 1],
        ])
        self.data = np.dot(self.data, rotation_matrix)

    def rotate_z(self, r):
        sin_theta = np.sin(r)
        cos_theta = np.cos(r)

        rotation_matrix = np.array([
            [cos_theta, -sin_theta, 0, 0],
            [sin_theta, cos_theta, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])
        self.data = np.dot(self.data, rotation_matrix)

    def scale(self, x, y, z):
        scaling_matrix = np.array([
            [x, 0, 0, 0],
            [0, y, 0, 0],
            [0, 0, z, 0],
            [0, 0, 0, 1]
        ])

        self.data = np.dot(self.data, scaling_matrix)

    @staticmethod
    def from_prs(position: v3f, rotation_euler: v3f, scale: v3f):
        m = Matrix4x4()
        m.translate(*position)
        m.rotate_z(rotation_euler[2])
        m.rotate_y(rotation_euler[1])
        m.rotate_x(rotation_euler[0])
        m.scale(*scale)
        return m

class Node:
    def __init__(self, position: v3f, rotation: v3f, scale: v3f, matrix: Matrix4x4, parent: int, label: str) -> None:
        self.position = position
        self.rotation = rotation
        self.scale = scale
        self.matrix = matrix
        self.parent = parent
        self.label = label

    def rotation_quat(self):
        rx = self.rotation[0]
        ry = self.rotation[1]
        rz = self.rotation[2]

        qx = np.sin(rx/2) * np.cos(ry/2) * np.cos(rz/2) - np.cos(rx/2) * np.sin(ry/2) * np.sin(rz/2)
        qy = np.cos(rx/2) * np.sin(ry/2) * np.cos(rz/2) + np.sin(rx/2) * np.cos(ry/2) * np.sin(rz/2)
        qz = np.cos(rx/2) * np.cos(ry/2) * np.sin(rz/2) - np.sin(rx/2) * np.sin(ry/2) * np.cos(rz/2)
        qw = np.cos(rx/2) * np.cos(ry/2) * np.cos(rz/2) + np.sin(rx/2) * np.sin(ry/2) * np.sin(rz/2)

        return (qx, qy, qz, qw)

class Color:
    def __init__(self, r: int = 255, g: int = 255, b: int = 255, a: int = 255) -> None:
        self.r = r
        self.g = g
        self.b = b
        self.a = a

class DataVertexVariation:
    def __init__(self) -> None:
        self.position = (0.0, 0.0, 0.0)
        self.weight = 0.0
        self.color = Color()
        self.normal = (0.0, 0.0, 0.0)
        self.idx_node = -1

class DataVertex:
    def __init__(self) -> None:
        self.num_variations = 0
        self.variations = [DataVertexVariation() for _ in range(4)]

class GfxVertexVariation:
    def __init__(self) -> None:
        self.position = (0.0, 0.0, 0.0)
        self.normal = (0.0, 0.0, 0.0)
        self.idx_node = 0

class GfxVertex:
    def __init__(self) -> None:
        self.variations = [GfxVertexVariation() for _ in range(4)]
        self.texcoord = (0.0, 0.0)
        self.color = (1.0, 1.0, 1.0, 1.0)
        self.weights = [0.0, 0.0, 0.0, 0.0]

class GfxStrip:
    def __init__(self) -> None:
        self.vertices: list[GfxVertex] = []
        self.indices: list[int] = []
        self.idx_texture = -1

class GfxPrimitive:
    def __init__(self, idx_node) -> None:
        self.idx_node = idx_node
        self.strips: list[GfxStrip] = []

class ProcessedVertexData:
    def __init__(self, position: v3f, normal: v3f) -> None:
        self.position = position
        self.normal = normal

def unpack_normal(xbr: XBEBinaryReader, virtual_address: int) -> v3f:
    def component_to_float(x):
        if x > 255:
            return -((511 - x) / 255)
        else:
            return x / 255
        
    packed_normal = xbr.read_u32(virtual_address)

    return (
        component_to_float((packed_normal >>  2) & 0x1ff),
        component_to_float((packed_normal >> 13) & 0x1ff),
        component_to_float((packed_normal >> 23) & 0x1ff)
    )

def normalize_v3f(v: v3f) -> v3f:
    v_length = math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])
    if v_length == 0.0:
        return (0.0, 0.0, 0.0)
    return (
        v[0] / v_length,
        v[1] / v_length,
        v[2] / v_length
    )

class RewindContext:
    def __init__(self, xbe: XBE) -> None:
        self.xbe = xbe
        self.xbr = XBEBinaryReader(xbe)

        self.current_node_index = 0
        self.nodes: list[Node] = []
        self.textures: list[tuple[str, PIL.Image.Image]] = []

        self.root_node_virtual_address = -1
        self.highest_num_variations = 1

        # BLiNX uses two separate vertex buffers, one for geometry and one for blendshapes.
        self.current_vertex_buffer_index = 0
        self.vertex_buffers = [
            [DataVertex() for _ in range(4096)],
            [DataVertex() for _ in range(4096)]
        ]

        # Used to calculate the final node/vertex position. Each time we parse a child node we go to next matrix in
        # stack. Index 0 is always an identity matrix.
        self.current_matrix_index = 0
        self.matrix_stack = [Matrix4x4() for _ in range(128)]

        # We use this value to try and find associated texture string table if one is not provided, since it often
        # appears above the model data in memory.
        self.lowest_referenced_virtual_address = 0xffffffff

        # While parsing triangles, some chunks may need to wait for vertices to be finished parsing. Therefore we need
        # to save addresses while parsing model data (see 'parsing_triangles'). This is how the game also handles it.
        self.saved_virtual_addresses = [0 for _ in range(256)]

        # This is what we'll eventually fill after parsing through the model data.
        self.primitives: list[GfxPrimitive] = []

    def parse_vertices(self, vaddr_block: int, allowed_variations: int):
        virtual_address = self.xbr.read_u32(vaddr_block + 0x00)
        if virtual_address == 0:
            return
        
        if virtual_address < self.lowest_referenced_virtual_address:
            self.lowest_referenced_virtual_address = virtual_address

        while self.xbr.read_u8(virtual_address) != 0xff:
            chunk_ident = self.xbr.read_u16(virtual_address)

            if (chunk_ident & 0xff) == 0x23:
                len_vertex = 0x10
                has_color = True
            elif (chunk_ident & 0xff) == 0x2c:
                len_vertex = 0x14
                has_color = False

                overwrite_vertices = (chunk_ident & 0x300) == 0
            else:
                len_vertex = 0x10
                has_color = False

            beg_vertices = self.xbr.read_u16(virtual_address + 0x04)
            num_vertices = self.xbr.read_u16(virtual_address + 0x06)

            # TODO: Differs in BLiNX 2; Add support for switching between the two.
            virtual_address += 0x10

            for i in range(num_vertices):
                position = self.xbr.read_v3f(virtual_address + 0x00)

                color = Color()
                normal = (0.0, 0.0, 0.0)

                if has_color:
                    color = Color(
                        self.xbr.read_u8(virtual_address + 0x0c + 0x02),
                        self.xbr.read_u8(virtual_address + 0x0c + 0x01),
                        self.xbr.read_u8(virtual_address + 0x0c + 0x00),
                        self.xbr.read_u8(virtual_address + 0x0c + 0x03)
                    )
                else:
                    normal = unpack_normal(self.xbr, virtual_address + 0x0c)

                if len_vertex == 0x14:
                    weight = self.xbr.read_u16(virtual_address + 0x10) / 255.0
                    idx_vertex = self.xbr.read_u16(virtual_address + 0x12)

                virtual_address += len_vertex

                if len_vertex == 0x14:
                    dst_vertex = self.vertex_buffers[self.current_vertex_buffer_index][beg_vertices + idx_vertex]
                    if overwrite_vertices:
                        dst_vertex.variations[0].position = position
                        dst_vertex.variations[0].weight = weight
                        dst_vertex.variations[0].color = color
                        dst_vertex.variations[0].normal = normal
                        dst_vertex.variations[0].idx_node = self.current_node_index
                        dst_vertex.num_variations = 1
                    else:
                        idx_variation = dst_vertex.num_variations
                        if idx_variation < allowed_variations:
                            dst_vertex.variations[idx_variation].position = position
                            dst_vertex.variations[idx_variation].weight = weight
                            dst_vertex.variations[idx_variation].color = color
                            dst_vertex.variations[idx_variation].normal = normal
                            dst_vertex.variations[idx_variation].idx_node = self.current_node_index
                            dst_vertex.num_variations += 1

                            if self.highest_num_variations < dst_vertex.num_variations:
                                self.highest_num_variations = dst_vertex.num_variations
                        else:
                            lowest_weight = 9999.0
                            lowest_weight_index = 0
                            total_weight = 0.0

                            if allowed_variations > 3:
                                for j in range(len(dst_vertex.variations)):
                                    total_weight += dst_vertex.variations[j].weight
                                    if dst_vertex.variations[j].weight < lowest_weight:
                                        lowest_weight = dst_vertex.variations[j].weight
                                        lowest_weight_index = j

                            if lowest_weight <= weight:
                                previous_weight = dst_vertex.variations[lowest_weight_index].weight

                                dst_vertex.variations[lowest_weight_index].position = position
                                dst_vertex.variations[lowest_weight_index].weight = weight
                                dst_vertex.variations[lowest_weight_index].color = color
                                dst_vertex.variations[lowest_weight_index].normal = normal
                                dst_vertex.variations[lowest_weight_index].idx_node = self.current_node_index

                                adjustment = (total_weight - previous_weight) + weight
                                adjustment = (previous_weight + adjustment) / adjustment

                                idx_variation = lowest_weight_index

                            if idx_variation < allowed_variations:
                                for j in range(allowed_variations):
                                    dst_vertex.variations[j].weight *= adjustment
                else:
                    dst_vertex = self.vertex_buffers[self.current_vertex_buffer_index][beg_vertices + i]
                    dst_vertex.variations[0].position = position
                    dst_vertex.variations[0].weight = 1.0
                    dst_vertex.variations[0].color = color
                    dst_vertex.variations[0].normal = normal
                    dst_vertex.variations[0].idx_node = self.current_node_index
                    dst_vertex.num_variations = 1

    def parse_triangles(self, vaddr_block: int):
        virtual_address = self.xbr.read_u32(vaddr_block + 0x04)
        if virtual_address == 0:
            return
        
        if virtual_address < self.lowest_referenced_virtual_address:
            self.lowest_referenced_virtual_address = virtual_address

        idx_texture = -1
        tint_color = Color()

        gfx_primitive = GfxPrimitive(self.current_node_index)
        while self.xbr.read_u8(virtual_address) != 0xff:
            chunk_ident = self.xbr.read_u16(virtual_address)

            # Padding
            if (chunk_ident & 0xff) == 0x00:
                virtual_address += 2

            # Save virtual address and break out of loop
            elif (chunk_ident & 0xff) == 0x04:
                virtual_address += 2
                self.saved_virtual_addresses[(chunk_ident & 0xff00) >> 8] = virtual_address
                break

            # Load virtual address
            elif (chunk_ident & 0xff) == 0x05:
                virtual_address = self.saved_virtual_addresses[(chunk_ident & 0xff00) >> 8]

            # Set current texture index
            elif (chunk_ident & 0xff) == 0x08:
                idx_texture = self.xbr.read_u8(virtual_address + 0x02)
                virtual_address += 4

            # Varied data
            elif (chunk_ident & 0xff) & 0x10:
                virtual_address += 0x04
                if (chunk_ident & 0xff) & 1:
                    tint_color = Color(
                        self.xbr.read_u8(virtual_address + 0x02),
                        self.xbr.read_u8(virtual_address + 0x01),
                        self.xbr.read_u8(virtual_address + 0x00),
                        self.xbr.read_u8(virtual_address + 0x03)
                    )
                    virtual_address += 0x04
                if (chunk_ident & 0xff) & 2:
                    # TODO: 4-bytes - color; Specular?
                    virtual_address += 0x04
                if (chunk_ident & 0xff) & 4:
                    # TODO: Unknown
                    virtual_address += 0x04
                if (chunk_ident & 0xff) & 8:
                    # TODO: Unknown
                    virtual_address += 0x0c

            # Triangle data
            elif (chunk_ident & 0xff) & 0x40:
                has_texcoord = (chunk_ident & 0xff) & 1

                num_strips = self.xbr.read_i16(virtual_address + 0x04) & 0x3fff
                virtual_address += 0x06

                
                for i in range(num_strips):
                    gfx_strip = GfxStrip()
                    gfx_strip.idx_texture = idx_texture if has_texcoord else -1

                    num_indices = self.xbr.read_i16(virtual_address + 0x00)
                    virtual_address += 2

                    # Negative number of indices indicates reverse triangle winding order.
                    inverse_winding_order = num_indices < 0
                    if num_indices < 0:
                        num_indices = -num_indices

                    for j in range(num_indices):
                        idx_vertex = self.xbr.read_u16(virtual_address)
                        virtual_address += 0x02

                        if has_texcoord:
                            texcoord = (
                                self.xbr.read_i16(virtual_address + 0x00) / 255.0,
                                self.xbr.read_i16(virtual_address + 0x02) / 255.0
                            )
                            virtual_address += 0x04
                        else:
                            texcoord = (0.0, 0.0)

                        vertex = self.vertex_buffers[self.current_vertex_buffer_index][idx_vertex]
                        gfx_vertex = GfxVertex()
                        for idx_variation in range(vertex.num_variations):
                            gfx_vertex.variations[idx_variation].position = vertex.variations[idx_variation].position
                            gfx_vertex.variations[idx_variation].normal = vertex.variations[idx_variation].normal
                            gfx_vertex.variations[idx_variation].idx_node = vertex.variations[idx_variation].idx_node
                            gfx_vertex.weights[idx_variation] = vertex.variations[idx_variation].weight

                        gfx_vertex.texcoord = texcoord
                        gfx_vertex.color = (
                            tint_color.r / 255.0,
                            tint_color.g / 255.0,
                            tint_color.b / 255.0,
                            tint_color.a / 255.0
                        )

                        gfx_strip.vertices.append(gfx_vertex)

                    # TODO: Respect doubled sided material flag
                    doubled_sided = chunk_ident & 0x1000

                    for j in range(num_indices - 2):
                        if (j % 2) == 0:
                            order = [1, 0, 2] if inverse_winding_order else [0, 1, 2]
                        else:
                            order = [0, 1, 2] if inverse_winding_order else [1, 0, 2]
                        gfx_strip.indices.extend([order[0] + j, order[1] + j, order[2] + j])

                    gfx_primitive.strips.append(gfx_strip)
            else:
                raise UnrecognizedChunkIdentifierError(chunk_ident, virtual_address - 2)

        self.primitives.append(gfx_primitive)

    def parse_block(self, virtual_address: int):
        if virtual_address < self.lowest_referenced_virtual_address:
            self.lowest_referenced_virtual_address = virtual_address

        allowed_variations = 4

        has_bs_vertices = False
        has_bs_triangles = False

        bs_block = None
        for bs in bs_blinx:
            if virtual_address == bs[0]:
                bs_block = bs[1]
                has_bs_vertices  = True
                has_bs_triangles = True
                break

        if bs_block == None:
            for bs in bs_001d4bd8:
                if virtual_address == bs[0]:
                    bs_block = bs[1]
                    has_bs_vertices = True
                    break

            for bs in bs_001d4e30:
                if virtual_address == bs[0]:
                    has_bs_triangles = True
                    break

        self.current_vertex_buffer_index = 0
        if has_bs_vertices or has_bs_triangles:
            allowed_variations = 3

        self.parse_vertices(virtual_address, allowed_variations)
        if has_bs_vertices:
            self.current_vertex_buffer_index = 1
            self.parse_vertices(bs_block, allowed_variations)

        self.current_vertex_buffer_index = 0
        self.parse_triangles(virtual_address)

        if has_bs_triangles:
            self.parse_triangles(bs_block)

    def parse_node(self, virtual_address: int, parent: int = -1):
        if virtual_address < self.lowest_referenced_virtual_address:
            self.lowest_referenced_virtual_address = virtual_address

        if parent == -1:
            self.root_node_virtual_address = virtual_address

        if self.current_matrix_index + 1 < 128:
            self.current_matrix_index += 1

        vaddr_block = self.xbr.read_u32(virtual_address + 0x04)
        position    = self.xbr.read_v3f(virtual_address + 0x08)
        rotation    = self.xbr.read_v3f(virtual_address + 0x14)
        scale       = self.xbr.read_v3f(virtual_address + 0x20)
        vaddr_child = self.xbr.read_u32(virtual_address + 0x2c)
        vaddr_next  = self.xbr.read_u32(virtual_address + 0x30)

        node_matrix = Matrix4x4.from_prs(position, rotation, scale)
        self.matrix_stack[self.current_matrix_index].data = np.dot(self.matrix_stack[self.current_matrix_index - 1].data, node_matrix.data)

        node_index = self.current_node_index
        self.nodes.append(Node(
            position = position,
            rotation = rotation,
            scale = scale,
            matrix = copy.deepcopy(self.matrix_stack[self.current_matrix_index]),
            parent = parent,
            label = f'Node_{node_index} @ {virtual_address:#0{10}x}'
        ))

        #print(self.nodes[node_index].label)

        if vaddr_block > 0:
            self.parse_block(vaddr_block)

        self.current_node_index += 1

        if vaddr_child > 0:
            self.parse_node(vaddr_child, node_index)

        if self.current_matrix_index > 0:
            self.current_matrix_index -= 1

        if vaddr_next > 0:
            self.parse_node(vaddr_next, parent)

    def parse_texture_list(self, virtual_address: int | None, texture_dirname: str, len_string: int):
        # If we've not supplied a virtual address to a string table, try and find matching string table by looking above
        # our lowest parsed virtual address in memory as the string table is often located above model data.
        if virtual_address == None:
            potential_vaddr_st = self.lowest_referenced_virtual_address - 0x08

            # Check that we're within bounds of an XBE section.
            if self.xbr.virtual_address_to_raw_address(potential_vaddr_st) != None and self.xbr.virtual_address_to_raw_address(potential_vaddr_st + 0x04) != None:
                potential_vaddr_strings = self.xbr.read_u32(potential_vaddr_st + 0x00)
                potential_num_strings = self.xbr.read_u32(potential_vaddr_st + 0x04)
                
                # If number of strings multiplied by string length brings us back to our potential string table virtual
                # address, then we most likely have a valid string table.
                if potential_num_strings > 0 and potential_vaddr_strings + (potential_num_strings * len_string) == potential_vaddr_st:
                    virtual_address = potential_vaddr_st

            if virtual_address == None:
                print('Could not find matching string table, please provide one via command-line arguments. Default materials will be applied to model.')
            else:
                print(f'Found matching string table at virtual address {virtual_address:#0{10}x}')

        if virtual_address != None:
            vaddr_entries = self.xbr.read_u32(virtual_address + 0x00)
            num_entries = self.xbr.read_u32(virtual_address + 0x04)

            for i in range(num_entries):
                filename = self.xbr.read_bytes(vaddr_entries + i * len_string, len_string).decode('ascii').rstrip('\x00')
                dds_filepath = os.path.join(texture_dirname, filename) + '.dds'
                try:
                    p_img = PIL.Image.open(dds_filepath)
                except PIL.UnidentifiedImageError:
                    print(f'Could not identify image format of DDS file ({dds_filepath}). You might need to apply this material manually, it will show as magenta colored.')
                    p_img = PIL.Image.new('RGB', (1, 1), (255, 0, 255))

                self.textures.append((filename, p_img))

# This function basically pseudo renders the vertex, taking in account of all the bones and weights. We can't submit
# this information directly to a format like glTF (which supports bones and weights) because the implementation of
# how we arrive at the final vertex positions differs between how it's done in BLiNX and most model formats. Luckily
# most of BLiNX's shader source code is uncompiled so it was relatively easy to figure out how the final vertex position
# is calculated.
# ARB instructions reference
# - https://www.downloads.redway3d.com/downloads/public/documentation/bk_bm_custom_custom_gpu_arb.html
def process_gfx_vertex(ctx: RewindContext, gfx_vertex: GfxVertex) -> ProcessedVertexData:
    def dph(r1, r2) -> float:
        return r1[0] * r2[0] + r1[1] * r2[1] + r1[2] * r2[2] + r2[3]

    def dp3(r1, r2) -> float:
        return r1[0] * r2[0] + r1[1] * r2[1] + r1[2] * r2[2]

    def mul(r1, r2) -> v4f:
        return (r1[0] * r2[0], r1[1] * r2[1], r1[2] * r2[2], r1[3] * r2[3])
    
    def mad(r1, r2, r3) -> v4f:
        return (r1[0] * r2[0] + r3[0], r1[1] * r2[1] + r3[1], r1[2] * r2[2] + r3[2], r1[3] * r2[3] + r3[3])

    def make_v4f(f) -> v4f:
        return (f, f, f, f)

    position = (0.0, 0.0, 0.0, 1.0)
    for idx_variation in range(4):
        variation = gfx_vertex.variations[idx_variation]

        md = ctx.nodes[variation.idx_node].matrix.data.T
        xyzw = (
            dph(variation.position, (md[0][0], md[1][0], md[2][0], md[3][0])),
            dph(variation.position, (md[0][1], md[1][1], md[2][1], md[3][1])),
            dph(variation.position, (md[0][2], md[1][2], md[2][2], md[3][2])),
            1.0
        )

        if idx_variation == 0:
            position = mul(make_v4f(gfx_vertex.weights[idx_variation]), xyzw)
        else:
            position = mad(make_v4f(gfx_vertex.weights[idx_variation]), xyzw, position)

    normal = (0.0, 0.0, 0.0, 1.0)
    for idx_variation in range(4):
        variation = gfx_vertex.variations[idx_variation]

        md = ctx.nodes[variation.idx_node].matrix.data.T
        xyzw = (
            dp3(variation.normal, (md[0][0], md[1][0], md[2][0], md[3][0])),
            dp3(variation.normal, (md[0][1], md[1][1], md[2][1], md[3][1])),
            dp3(variation.normal, (md[0][2], md[1][2], md[2][2], md[3][2])),
            1.0
        )

        if idx_variation == 0:
            normal = mul(make_v4f(gfx_vertex.weights[idx_variation]), xyzw)
        else:
            normal = mad(make_v4f(gfx_vertex.weights[idx_variation]), xyzw, normal)

    def xyz(v) -> v3f:
        return (v[0], v[1], v[2])

    return ProcessedVertexData(xyz(position), xyz(normal))

def save_as_gltf(ctx: RewindContext, output_directory: str, format: Literal['gltf', 'glb'], output_basename: str | None):
    treat_as_character = ctx.highest_num_variations > 1

    gltf_model = gltflib.GLTFModel(
        asset = gltflib.Asset(generator='Rewind (https://github.com/itsybitsypixel/rewind)')
    )

    def gltf_add_node(gltf_node: gltflib.Node):
        if gltf_model.nodes == None:
            gltf_model.nodes = []

        gltf_model.nodes.append(gltf_node)
        return len(gltf_model.nodes) - 1
    
    def gltf_add_buffer_view(gltf_buffer_view: gltflib.BufferView):
        if gltf_model.bufferViews == None:
            gltf_model.bufferViews = []

        gltf_model.bufferViews.append(gltf_buffer_view)
        return len(gltf_model.bufferViews) - 1
    
    def gltf_add_child(idx_parent: int, idx_child: int):
        if gltf_model.nodes[idx_parent].children == None:
            gltf_model.nodes[idx_parent].children = []

        gltf_model.nodes[idx_parent].children.append(idx_child)

    def gltf_add_accessor(gltf_accessor: gltflib.Accessor):
        if gltf_model.accessors == None:
            gltf_model.accessors = []

        gltf_model.accessors.append(gltf_accessor)
        return len(gltf_model.accessors) - 1

    def gltf_add_mesh(gltf_mesh: gltflib.Mesh):
        if gltf_model.meshes == None:
            gltf_model.meshes = []

        gltf_model.meshes.append(gltf_mesh)
        return len(gltf_model.meshes) - 1

    def gltf_add_scene(gltf_scene: gltflib.Scene):
        if gltf_model.scenes == None:
            gltf_model.scenes = []

        gltf_model.scenes.append(gltf_scene)
        return len(gltf_model.scenes) - 1
    
    def gltf_add_buffer(gltf_buffer: gltflib.Buffer):
        if gltf_model.buffers == None:
            gltf_model.buffers = []

        gltf_model.buffers.append(gltf_buffer)
        return len(gltf_model.buffers) - 1

    def gltf_add_image(gltf_image: gltflib.Image):
        if gltf_model.images == None:
            gltf_model.images = []

        gltf_model.images.append(gltf_image)
        return len(gltf_model.images) - 1

    def gltf_add_texture(gltf_texture: gltflib.Texture):
        if gltf_model.textures == None:
            gltf_model.textures = []

        gltf_model.textures.append(gltf_texture)
        return len(gltf_model.textures) - 1

    def gltf_add_material(gltf_material: gltflib.Material):
        if gltf_model.materials == None:
            gltf_model.materials = []

        gltf_model.materials.append(gltf_material)
        return len(gltf_model.materials) - 1

    gltf_data = bytearray()
    def gltf_pad_data():
        gltf_data.extend(b'\xcc' * (16 - (len(gltf_data) % 16)))

    # Check if we'll need a default material and what our highest texture index is.
    need_default_material = False
    highest_texture_index = -1
    for gfx_primitive in ctx.primitives:
        for gfx_strip in gfx_primitive.strips:
            if gfx_strip.idx_texture == -1:
                need_default_material = True

            if gfx_strip.idx_texture > highest_texture_index:
                highest_texture_index = gfx_strip.idx_texture

    idx_texture_to_gltf_material = {}
    if need_default_material:
        gltf_default_material = gltf_add_material(gltflib.Material(
            name = 'Default'
        ))

        idx_texture_to_gltf_material[-1] = gltf_default_material

    # If we have not found/supplied any textures, fill with empty materials so we can still differentiate and manually
    # apply materials.
    if len(ctx.textures) == 0:
        for i in range(highest_texture_index + 1):
            idx_texture_to_gltf_material[i] = gltf_add_material(gltflib.Material(
                name = f'Material_{i}',
                doubleSided = True
            ))

    for idx_texture, (img_filename, img) in enumerate(ctx.textures):
        gltf_pad_data()

        tmp_io = io.BytesIO()
        img.save(tmp_io, format='png')

        gltf_bv_image = gltf_add_buffer_view(gltflib.BufferView(
            buffer = 0,
            byteLength = tmp_io.getbuffer().nbytes,
            byteOffset = len(gltf_data)
        ))

        gltf_data.extend(tmp_io.getvalue())

        gltf_image = gltf_add_image(gltflib.Image(
            name = f'{img_filename}.img',
            bufferView = gltf_bv_image,
            mimeType = 'image/png'
        ))

        gltf_texture = gltf_add_texture(gltflib.Texture(
            source = gltf_image
        ))

        idx_texture_to_gltf_material[idx_texture] = gltf_add_material(gltflib.Material(
            name = img_filename,
            pbrMetallicRoughness = gltflib.PBRMetallicRoughness(
                baseColorTexture = gltflib.TextureInfo(
                    index = gltf_texture
                )
            )
        ))

    for node in ctx.nodes:
        gltf_idx_node = gltf_add_node(gltflib.Node(
            name = node.label,
            translation = node.position,
            rotation = node.rotation_quat(),
            scale = node.scale
        ))

        if node.parent >= 0:
            gltf_add_child(node.parent, gltf_idx_node)

    node_gltf_primitives = {}
    for gfx_primitive in ctx.primitives:
        for gfx_strip in gfx_primitive.strips:
            gltf_pad_data()

            has_texture = gfx_strip.idx_texture >= 0

            if has_texture:
                gltf_len_vertex = 0x30
            else:
                gltf_len_vertex = 0x28

            gltf_bv_vertices = gltf_add_buffer_view(gltflib.BufferView(
                buffer = 0,
                byteOffset = len(gltf_data),
                byteStride = gltf_len_vertex,
                byteLength = gltf_len_vertex * len(gfx_strip.vertices),
                target = gltflib.BufferTarget.ARRAY_BUFFER.value
            ))

            v_positions = []
            for gfx_vertex in gfx_strip.vertices:
                if treat_as_character:
                    processed_vertex = process_gfx_vertex(ctx, gfx_vertex)
                    v3f_position = processed_vertex.position
                    v3f_normal = processed_vertex.normal
                else:
                    v3f_position = gfx_vertex.variations[0].position
                    v3f_normal = gfx_vertex.variations[0].normal

                v_positions.append(v3f_position)
                v3f_normal = normalize_v3f(v3f_normal)

                gltf_data.extend(struct.pack('3f', *v3f_position))
                gltf_data.extend(struct.pack('3f', *v3f_normal))
                if has_texture:
                    gltf_data.extend(struct.pack('2f', *gfx_vertex.texcoord))
                gltf_data.extend(struct.pack('4f', *gfx_vertex.color))

            gltf_bv_indices = gltf_add_buffer_view(gltflib.BufferView(
                buffer = 0,
                byteOffset = len(gltf_data),
                byteLength = 2 * len(gfx_strip.indices),
                target = gltflib.BufferTarget.ELEMENT_ARRAY_BUFFER.value
            ))

            for i in range(len(gfx_strip.indices)):
                gltf_data.extend(struct.pack('<H', gfx_strip.indices[i]))

            byte_offset = 0
            gltf_accessor_position = gltf_add_accessor(gltflib.Accessor(
                byteOffset = byte_offset,
                count = len(gfx_strip.vertices),
                bufferView = gltf_bv_vertices,
                componentType = gltflib.ComponentType.FLOAT.value,
                type = gltflib.AccessorType.VEC3.value,
                min=tuple(min(v_positions,key=operator.itemgetter(i))[i] for i in range(3)),
                max=tuple(max(v_positions,key=operator.itemgetter(i))[i] for i in range(3))
            ))
            byte_offset += 12

            gltf_accessor_normal = gltf_add_accessor(gltflib.Accessor(
                byteOffset = byte_offset,
                count = len(gfx_strip.vertices),
                bufferView = gltf_bv_vertices,
                componentType = gltflib.ComponentType.FLOAT.value,
                type = gltflib.AccessorType.VEC3.value
            ))
            byte_offset += 12

            if has_texture:
                gltf_accessor_texcoord = gltf_add_accessor(gltflib.Accessor(
                    byteOffset = byte_offset,
                    count = len(gfx_strip.vertices),
                    bufferView = gltf_bv_vertices,
                    componentType = gltflib.ComponentType.FLOAT.value,
                    type = gltflib.AccessorType.VEC2.value
                ))
                byte_offset += 8

            gltf_accessor_color = gltf_add_accessor(gltflib.Accessor(
                byteOffset = byte_offset,
                count = len(gfx_strip.vertices),
                bufferView = gltf_bv_vertices,
                componentType = gltflib.ComponentType.FLOAT.value,
                type = gltflib.AccessorType.VEC4.value
            ))
            byte_offset += 16

            gltf_accessor_indices = gltf_add_accessor(gltflib.Accessor(
                byteOffset = 0,
                count = len(gfx_strip.indices),
                bufferView = gltf_bv_indices,
                componentType = gltflib.ComponentType.UNSIGNED_SHORT.value,
                type = gltflib.AccessorType.SCALAR.value
            ))

            if gfx_primitive.idx_node not in node_gltf_primitives:
                node_gltf_primitives[gfx_primitive.idx_node] = []

            node_gltf_primitives[gfx_primitive.idx_node].append(gltflib.Primitive(
                mode = gltflib.PrimitiveMode.TRIANGLES.value,
                material = idx_texture_to_gltf_material[gfx_strip.idx_texture],
                indices = gltf_accessor_indices,
                attributes = gltflib.Attributes(
                    POSITION = gltf_accessor_position,
                    NORMAL = gltf_accessor_normal,
                    TEXCOORD_0 = gltf_accessor_texcoord if has_texture else None,
                    COLOR_0 = gltf_accessor_color,
                )
            ))
    
    if treat_as_character:
        gltf_all_primitives = []
        for gltf_primitives in node_gltf_primitives.values():
            gltf_all_primitives = gltf_all_primitives + gltf_primitives

        gltf_mesh = gltf_add_mesh(gltflib.Mesh(
            name = 'skinned_mesh',
            primitives = gltf_all_primitives
        ))
        
        gltf_model.nodes[0].mesh = gltf_mesh
    else:
        for idx_node, gltf_primitives in node_gltf_primitives.items():
            gltf_mesh = gltf_add_mesh(gltflib.Mesh(
                name = f'{gltf_model.nodes[idx_node].name}.mesh',
                primitives = gltf_primitives
            ))

            gltf_model.nodes[idx_node].mesh = gltf_mesh

    gltf_model.scene = gltf_add_scene(gltflib.Scene(
        nodes = [0]
    ))

    uri = f'{ctx.root_node_virtual_address:#0{10}x}.bin'

    if output_basename:
        output_filename = f'{output_basename}.{format}'
        uri = f'{output_basename}.bin'
    else:
        output_filename = f'{ctx.root_node_virtual_address:#0{10}x}.{format}'
        uri = f'{ctx.root_node_virtual_address:#0{10}x}.bin'

    gltf_add_buffer(gltflib.Buffer(
        byteLength = len(gltf_data),
        uri = uri
    ))

    gltflib.GLTF(
        gltf_model,
        [gltflib.FileResource(
            filename = uri,
            data = gltf_data
        )]
    ).export(os.path.join(output_directory, output_filename))

if __name__ == '__main__':
    supported_formats = [
        'gltf',
        'glb'
    ]

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('xbe_filename')
    args_parser.add_argument('-n', '--node-vaddr', 
                             required = True,
                             help = 'virtual address of root node')
    
    args_parser.add_argument('-s', '--st-vaddr',
                             help = 'virtual address of string table containing texture names')
    
    args_parser.add_argument('-t', '--texture-directory',
                             help = 'directory where game textures are stored',
                             required = True)
    
    args_parser.add_argument('-f', '--format',
                             help = 'output format; supported formats = [' + ', '.join(supported_formats) + ']',
                             default = 'glb')

    args_parser.add_argument('-b', '--basename',
                             help = 'specify a basename for output file(s); if not specified, file will have basename of the starting node\'s virtual address')

    args_parser.add_argument('-o', '--output-directory',
                             required = True)
    
    args = args_parser.parse_args()

    args.format = args.format.lower()
    if args.format not in supported_formats:
        print(f'Unsupported output format \'{args.format}\', supported formats are [' + ', '.join(supported_formats) + ']')
        sys.exit(1)

    if len(args.node_vaddr) > 2 and args.node_vaddr[0:2] == '0x':
        args.node_vaddr = int(args.node_vaddr, 16)
    else:
        args.node_vaddr = int(args.node_vaddr)

    xbe = XBE.from_filepath(args.xbe_filename)

    ctx = RewindContext(xbe)
    ctx.parse_node(args.node_vaddr)

    if args.st_vaddr == None:
        print(f'No virtual address provided for texture string table, will try finding matching string table...')

    ctx.parse_texture_list(args.st_vaddr, args.texture_directory, 32)

    if args.format in ['gltf', 'glb']:
        save_as_gltf(ctx, args.output_directory, args.format, args.basename)
